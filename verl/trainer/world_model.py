"""
World Model training module for LLM-based world model.
The LLM acts as both encoder and transition model.
LLM generates latent tokens in format: <LATENT> Z_1 Z_57 Z_203 ... </LATENT>
Loss is computed as CrossEntropy between predicted and target tokens (language modeling loss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np
import re
from verl import DataProto
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


class WorldModelTrainer:
    """
    Trainer for LLM-based world model.
    
    The world model consists of:
    1. Encoder: LLM encodes observation ot -> latent tokens zt = <LATENT> Z_1 Z_57 ... </LATENT>
    2. Transition: LLM predicts next state \hat z_{t+1} from (ot, zt, at)
    3. Target: LLM encodes ot+1 -> z'_{t+1} (without gradients)
    4. Loss: CrossEntropy between predicted tokens \hat z_{t+1} and target tokens z'_{t+1}
    """
    
    def __init__(
        self,
        actor_rollout_wg,
        tokenizer,
        encoder_prompt: str = "Преобразуй наблюдение среды во внутреннее латентное состояние.\nВерни ТОЛЬКО латентные токены в формате:\n<LATENT> ... </LATENT>",
        transition_prompt: str = "Предскажи следующее латентное состояние среды.",
        max_latent_tokens: int = 64,  # Maximum number of latent tokens to generate
        device: str = "cuda",
    ):
        """
        Args:
            actor_rollout_wg: Actor rollout worker group (contains the LLM model)
            tokenizer: Tokenizer for the LLM
            encoder_prompt: Prompt for encoder mode
            transition_prompt: Prompt for transition model mode
            max_latent_tokens: Maximum number of latent tokens to generate
            device: Device to run on
        """
        self.actor_rollout_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self.encoder_prompt = encoder_prompt
        self.transition_prompt = transition_prompt
        self.max_latent_tokens = max_latent_tokens
        self.device = device
        
        # Special tokens for latent representation
        self.latent_start_token = "<LATENT>"
        self.latent_end_token = "</LATENT>"
        
        # Generation config for latent token generation
        self.max_new_tokens = max_latent_tokens + 10  # Extra tokens for special tokens and formatting
    
    def _extract_latent_tokens(self, text: str) -> List[int]:
        """
        Extract latent tokens from LLM-generated text.
        
        Expected format: <LATENT> Z_1 Z_57 Z_203 ... </LATENT>
        or any text containing <LATENT>...</LATENT> block.
        
        Args:
            text: Generated text from LLM
            
        Returns:
            List of token IDs representing the latent state
        """
        # Find content between <LATENT> and </LATENT>
        pattern = rf'{re.escape(self.latent_start_token)}(.*?){re.escape(self.latent_end_token)}'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            latent_content = match.group(1).strip()
            # Tokenize the content between tags
            latent_tokens = self.tokenizer.encode(latent_content, add_special_tokens=False)
            return latent_tokens
        else:
            # If no <LATENT> tags found, try to extract tokens from the whole text
            # This is a fallback - ideally LLM should generate proper format
            # Tokenize the entire response (might contain extra text)
            all_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            # Limit to max_latent_tokens
            return all_tokens[:self.max_latent_tokens]
    
    def _generate_latent_tokens(
        self,
        observations: list,
        prompt_template: str,
        with_grad: bool = True,
    ) -> List[List[int]]:
        """
        Generate latent tokens from LLM.
        
        Args:
            observations: List of observation strings
            prompt_template: Prompt template to use
            with_grad: Whether to compute gradients
            
        Returns:
            List of lists of token IDs for each observation
        """
        batch_size = len(observations)
        
        # Create prompts
        if prompt_template:
            prompts = [f"{prompt_template}\n{obs}" for obs in observations]
        else:
            prompts = observations
        
        # Format as chat messages
        chat_prompts = []
        for prompt in prompts:
            chat = [{"role": "user", "content": prompt}]
            chat_prompts.append(chat)
        
        # Apply chat template and tokenize
        batch_dict = {
            'input_ids': [],
            'attention_mask': [],
            'position_ids': [],
        }
        non_tensor_batch = {
            'raw_prompt': chat_prompts,
        }
        
        for chat in chat_prompts:
            # Apply chat template
            prompt_text = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Tokenize
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_text,
                tokenizer=self.tokenizer,
                max_length=2048,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='error',
            )
            
            position_ids = compute_position_id_with_mask(attention_mask)
            
            batch_dict['input_ids'].append(input_ids[0])
            batch_dict['attention_mask'].append(attention_mask[0])
            batch_dict['position_ids'].append(position_ids[0])
        
        # Convert to tensors
        batch_dict['input_ids'] = torch.stack(batch_dict['input_ids'])
        batch_dict['attention_mask'] = torch.stack(batch_dict['attention_mask'])
        batch_dict['position_ids'] = torch.stack(batch_dict['position_ids'])
        
        # Create DataProto
        gen_batch = DataProto.from_dict(
            tensors=batch_dict,
            non_tensors=non_tensor_batch,
            meta_info={}
        )
        
        # Set generation parameters
        gen_batch.meta_info = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.1,  # Low temperature for more deterministic output
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Generate text from LLM
        if with_grad:
            output = self.actor_rollout_wg.generate_sequences(gen_batch)
        else:
            with torch.no_grad():
                output = self.actor_rollout_wg.generate_sequences(gen_batch)
        
        # Decode responses
        responses = output.batch['responses']  # (batch_size, seq_len)
        response_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        
        # Extract latent tokens from text
        latent_tokens_list = []
        for text in response_texts:
            latent_tokens = self._extract_latent_tokens(text)
            latent_tokens_list.append(latent_tokens)
        
        return latent_tokens_list
    
    def encode_observation(
        self,
        observations: list,
        with_grad: bool = True,
    ) -> List[List[int]]:
        """
        Encode observations to latent tokens using LLM as encoder.
        
        Args:
            observations: List of observation strings
            with_grad: Whether to compute gradients
            
        Returns:
            List of lists of latent token IDs
        """
        return self._generate_latent_tokens(
            observations=observations,
            prompt_template=self.encoder_prompt,
            with_grad=with_grad,
        )
    
    def predict_next_state(
        self,
        observations: list,
        current_latent_tokens: List[List[int]],
        actions: list,
        with_grad: bool = True,
    ) -> List[List[int]]:
        """
        Predict next state latent tokens using LLM as transition model.
        
        Args:
            observations: List of current observation strings
            current_latent_tokens: List of lists of current latent token IDs
            actions: List of action strings
            with_grad: Whether to compute gradients
            
        Returns:
            List of lists of predicted next state latent token IDs
        """
        batch_size = len(observations)
        
        # Create prompts with current latent tokens
        prompts = []
        for i in range(batch_size):
            # Decode current latent tokens to text for the prompt
            current_latent_text = self.tokenizer.decode(current_latent_tokens[i], skip_special_tokens=False)
            prompt = f"{self.transition_prompt}\nНаблюдение: {observations[i]}\nТекущее латентное состояние: {current_latent_text}\nДействие: {actions[i]}"
            prompts.append(prompt)
        
        return self._generate_latent_tokens(
            observations=prompts,
            prompt_template="",  # Already included in prompts
            with_grad=with_grad,
        )
    
    def _compute_language_modeling_loss(
        self,
        predicted_tokens: List[List[int]],
        target_tokens: List[List[int]],
        input_prompts: List[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CrossEntropy loss between predicted and target tokens.
        
        This uses compute_log_prob to get log probabilities of target tokens
        given the input prompts, then computes negative log likelihood.
        
        Args:
            predicted_tokens: List of lists of predicted token IDs (not used directly, but for reference)
            target_tokens: List of lists of target token IDs
            input_prompts: List of input prompt strings (transition prompts with ot, zt, at)
            
        Returns:
            loss: Language modeling loss (negative log likelihood)
            metrics: Dictionary of metrics
        """
        batch_size = len(target_tokens)
        
        # Tokenize prompts and create input sequences
        batch_dict = {
            'input_ids': [],
            'attention_mask': [],
            'position_ids': [],
            'responses': [],  # Target tokens (what we want to compute log_probs for)
        }
        
        # Find max lengths for padding
        prompt_lengths = []
        target_lengths = []
        for prompt, target_toks in zip(input_prompts, target_tokens):
            # Apply chat template to prompt
            chat = [{"role": "user", "content": prompt}]
            prompt_text = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False
            )
            prompt_ids, _ = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_text,
                tokenizer=self.tokenizer,
                max_length=2048,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='error',
            )
            prompt_lengths.append(prompt_ids[0].shape[0])
            target_lengths.append(len(target_toks))
        
        max_prompt_len = max(prompt_lengths)
        max_target_len = max(target_lengths)
        max_total_len = max_prompt_len + max_target_len
        
        for i, (prompt, target_toks) in enumerate(zip(input_prompts, target_tokens)):
            # Apply chat template and tokenize prompt
            chat = [{"role": "user", "content": prompt}]
            prompt_text = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False
            )
            prompt_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_text,
                tokenizer=self.tokenizer,
                max_length=2048,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='error',
            )
            prompt_ids = prompt_ids[0]  # Remove batch dimension
            attention_mask = attention_mask[0]
            
            # Pad prompt to max_prompt_len
            prompt_len = prompt_ids.shape[0]
            if prompt_len < max_prompt_len:
                padding = torch.full((max_prompt_len - prompt_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                prompt_ids = torch.cat([prompt_ids, padding])
                attention_mask = torch.cat([attention_mask, torch.zeros(max_prompt_len - prompt_len, dtype=torch.long)])
            
            batch_dict['input_ids'].append(prompt_ids)
            batch_dict['attention_mask'].append(attention_mask)
            
            # Target tokens (responses)
            target_tensor = torch.tensor(target_toks, dtype=torch.long)
            if len(target_tensor) < max_target_len:
                padding = torch.full((max_target_len - len(target_tensor),), self.tokenizer.pad_token_id, dtype=torch.long)
                target_tensor = torch.cat([target_tensor, padding])
            batch_dict['responses'].append(target_tensor)
        
        # Stack tensors
        batch_dict['input_ids'] = torch.stack(batch_dict['input_ids'])
        batch_dict['attention_mask'] = torch.stack(batch_dict['attention_mask'])
        batch_dict['position_ids'] = compute_position_id_with_mask(batch_dict['attention_mask'])
        batch_dict['responses'] = torch.stack(batch_dict['responses'])
        
        # Create DataProto
        data_batch = DataProto.from_dict(
            tensors=batch_dict,
            non_tensors={},
            meta_info={}
        )
        
        # Compute log probabilities of target tokens given prompts
        # This computes: log P(target_tokens | prompt)
        log_prob_output = self.actor_rollout_wg.compute_log_prob(data_batch)
        log_probs = log_prob_output.batch['log_probs']  # (batch_size, response_length)
        
        # Create mask for valid tokens (non-padding)
        response_mask = (batch_dict['responses'] != self.tokenizer.pad_token_id).float()
        
        # Compute negative log likelihood (language modeling loss)
        # Loss = -mean(log_probs) over valid tokens
        # This is equivalent to CrossEntropy loss
        masked_log_probs = log_probs * response_mask
        valid_tokens = response_mask.sum().clamp(min=1)
        loss = -masked_log_probs.sum() / valid_tokens
        
        # Compute metrics
        with torch.no_grad():
            per_sample_valid = response_mask.sum(dim=1).clamp(min=1)
            per_sample_loss = -(masked_log_probs.sum(dim=1) / per_sample_valid)
            
            metrics = {
                "world_model/loss": loss.item(),
                "world_model/loss_mean": per_sample_loss.mean().item(),
                "world_model/loss_std": per_sample_loss.std().item(),
                "world_model/avg_log_prob": (masked_log_probs.sum() / valid_tokens).item(),
                "world_model/avg_target_length": per_sample_valid.mean().item(),
            }
        
        return loss, metrics
    
    def compute_loss(
        self,
        observations: list,
        next_observations: list,
        actions: list,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute world model loss.
        
        Args:
            observations: List of current observation strings
            next_observations: List of next observation strings
            actions: List of action strings
            
        Returns:
            loss: World model loss (CrossEntropy between predicted and target tokens)
            metrics: Dictionary of metrics
        """
        # Step 1: Encode current observations (with gradients)
        current_latent_tokens = self.encode_observation(observations, with_grad=True)
        
        # Step 2: Predict next state (with gradients)
        predicted_next_latent_tokens = self.predict_next_state(
            observations=observations,
            current_latent_tokens=current_latent_tokens,
            actions=actions,
            with_grad=True,
        )
        
        # Step 3: Encode next observations (without gradients) - target
        target_next_latent_tokens = self.encode_observation(next_observations, with_grad=False)
        
        # Step 4: Create prompts for transition model (for computing log_probs)
        transition_prompts = []
        for i in range(len(observations)):
            current_latent_text = self.tokenizer.decode(current_latent_tokens[i], skip_special_tokens=False)
            prompt = f"{self.transition_prompt}\nНаблюдение: {observations[i]}\nТекущее латентное состояние: {current_latent_text}\nДействие: {actions[i]}"
            transition_prompts.append(prompt)
        
        # Step 5: Compute CrossEntropy loss (language modeling loss)
        loss, metrics = self._compute_language_modeling_loss(
            predicted_tokens=predicted_next_latent_tokens,
            target_tokens=target_next_latent_tokens,
            input_prompts=transition_prompts,
        )
        
        return loss, metrics
    
    def train_step(
        self,
        observations: list,
        next_observations: list,
        actions: list,
        optimizer: torch.optim.Optimizer,
        loss_coef: float = 1.0,
    ) -> Dict[str, float]:
        """
        Perform one training step for the world model.
        
        Args:
            observations: List of current observation strings
            next_observations: List of next observation strings
            actions: List of action strings
            optimizer: Optimizer for the world model (should optimize actor_rollout_wg model)
            loss_coef: Coefficient for world model loss (for combining with PPO loss)
            
        Returns:
            metrics: Dictionary of metrics
        """
        try:
            # Note: We don't call optimizer.zero_grad() here because
            # it should be called before PPO update, and we share the same optimizer
            # The world model loss will be added to the computation graph
            
            loss, metrics = self.compute_loss(
                observations=observations,
                next_observations=next_observations,
                actions=actions,
            )
            
            # Scale loss by coefficient
            scaled_loss = loss * loss_coef
            
            # Backward pass (gradients accumulate with PPO gradients)
            scaled_loss.backward()
            
            # Note: We don't call optimizer.step() here because
            # it should be called after both PPO and world model losses are computed
            # The optimizer.step() is called in the PPO update step
            
            # Update metrics with scaled loss
            metrics["world_model/scaled_loss"] = scaled_loss.item()
            metrics["world_model/loss_coef"] = loss_coef
            
            # Debug: ensure metrics are not empty
            if not metrics:
                print("[World Model] Warning: compute_loss returned empty metrics dict")
            else:
                print(f"[World Model] train_step returning {len(metrics)} metrics: {list(metrics.keys())}")
            
            return metrics
        except Exception as e:
            print(f"[World Model] Error in train_step: {e}")
            import traceback
            traceback.print_exc()
            return {}
