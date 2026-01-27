"""
Утилита для извлечения позиций токенов action из response текста.
Используется для вычисления log_prob только для токенов внутри <action> тегов.
"""
import re
import torch
from typing import List, Tuple, Optional


def extract_action_token_positions(
    response_texts: List[str],
    response_token_ids: torch.Tensor,
    tokenizer
) -> torch.Tensor:
    """
    Извлекает маску для токенов, которые находятся внутри <action>...</action> тегов.
    
    Args:
        response_texts: Список текстовых ответов (batch_size)
        response_token_ids: Тензор токенов response [batch_size, response_length]
        tokenizer: Токенизатор для декодирования токенов
        
    Returns:
        action_mask: Тензор булевых масок [batch_size, response_length], 
                    где True означает, что токен находится внутри <action> тегов
    """
    batch_size, response_length = response_token_ids.shape
    action_mask = torch.zeros(batch_size, response_length, dtype=torch.bool, device=response_token_ids.device)
    
    for i, response_text in enumerate(response_texts):
        # Находим позиции <action> и </action> в тексте
        start_tag = "<action>"
        end_tag = "</action>"
        
        start_idx = response_text.lower().find(start_tag.lower())
        end_idx = response_text.lower().find(end_tag.lower())
        
        if start_idx == -1 or end_idx == -1:
            # Если тегов нет, маска остается нулевой (все False)
            continue
        
        # Извлекаем текст внутри тегов (включая сами теги для точного сопоставления)
        action_start_text_pos = start_idx
        action_end_text_pos = end_idx + len(end_tag)
        
        # Декодируем response_token_ids обратно в текст для точного сопоставления
        # Это нужно для нахождения точных позиций токенов
        response_tokens = response_token_ids[i].cpu().tolist()
        
        # Находим позиции токенов, которые соответствуют action части
        # Используем посимвольное сопоставление
        decoded_so_far = ""
        token_start_positions = []
        
        for token_idx, token_id in enumerate(response_tokens):
            # Декодируем токен
            try:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            except:
                token_text = ""
            
            # Сохраняем позицию начала этого токена в декодированном тексте
            token_start_positions.append(len(decoded_so_far))
            decoded_so_far += token_text
            
            # Если мы достигли конца response, останавливаемся
            if len(decoded_so_far) >= len(response_text):
                break
        
        # Теперь находим, какие токены попадают в диапазон action
        for token_idx, token_start_pos in enumerate(token_start_positions):
            if token_idx >= response_length:
                break
                
            # Декодируем токен для определения его длины
            try:
                token_text = tokenizer.decode([response_tokens[token_idx]], skip_special_tokens=False)
                token_end_pos = token_start_pos + len(token_text)
            except:
                continue
            
            # Проверяем, пересекается ли токен с action частью
            # Токен попадает в action, если его позиция пересекается с [action_start_text_pos, action_end_text_pos]
            if (token_start_pos < action_end_text_pos and token_end_pos > action_start_text_pos):
                action_mask[i, token_idx] = True
    
    return action_mask


def extract_action_token_positions_simple(
    response_texts: List[str],
    response_token_ids: torch.Tensor,
    tokenizer
) -> torch.Tensor:
    """
    Упрощенная версия: извлекает маску для токенов внутри <action> тегов.
    Использует более простой подход - декодирует весь response и находит позиции.
    
    Args:
        response_texts: Список текстовых ответов (batch_size)
        response_token_ids: Тензор токенов response [batch_size, response_length]
        tokenizer: Токенизатор для декодирования токенов
        
    Returns:
        action_mask: Тензор булевых масок [batch_size, response_length]
    """
    batch_size, response_length = response_token_ids.shape
    action_mask = torch.zeros(batch_size, response_length, dtype=torch.bool, device=response_token_ids.device)
    
    for i, response_text in enumerate(response_texts):
        # Находим позиции <action> и </action> в тексте
        start_tag = "<action>"
        end_tag = "</action>"
        
        start_idx = response_text.lower().find(start_tag.lower())
        end_idx = response_text.lower().find(end_tag.lower())
        
        if start_idx == -1 or end_idx == -1:
            continue
        
        # Декодируем токены по одному и находим, какие попадают в action
        decoded_pos = 0
        action_start_char = start_idx
        action_end_char = end_idx + len(end_tag)
        
        for token_idx in range(response_length):
            if token_idx >= response_token_ids.shape[1]:
                break
                
            token_id = response_token_ids[i, token_idx].item()
            try:
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            except:
                token_text = ""
            
            token_start_char = decoded_pos
            token_end_char = decoded_pos + len(token_text)
            
            # Проверяем, попадает ли токен в action часть
            if token_start_char < action_end_char and token_end_char > action_start_char:
                action_mask[i, token_idx] = True
            
            decoded_pos = token_end_char
            
            # Если мы прошли весь текст, останавливаемся
            if decoded_pos >= len(response_text):
                break
    
    return action_mask
