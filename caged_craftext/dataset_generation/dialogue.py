from openai import OpenAI

import os
from tqdm import tqdm
import argparse
from prompts_generator import generate_prompts


def check_and_format(content):
    """Extracts content after the first '=' if present."""
    if "=" in content:
        return content.split("=", 1)[1]
    return content


def main(count_goals, instructions_class, difficulty, output_file, client):
    """Generates instructions based on prompts."""
    instructions = []
    print("Generating promts")
    prompts = generate_prompts(count_goals, instructions_class, difficulty)
   # print(prompts[0])
    print("Promts generated")
    
    for prompt in tqdm(prompts, desc='Generating Instructions'):
        response = client.chat.completions.create(model="gpt-4",
        messages=[{"role": "user", "content": prompt}])
        content = response.choices[0].message.content
        content = check_and_format(content)
        instructions.append(content)

        with open(output_file, "a") as f:
            f.write(content + "\n----\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate Instructions from Prompts')
    parser.add_argument('--count_goals',        type=int, required=True, help='Number of goals to generate instructions for')
    parser.add_argument('--instructions_class', type=str, required=True, help='Class of instructions to generate: achivments|building_line|building_squere|conditonal_placing|localization_placing')
    parser.add_argument('--output_file',        type=str, required=True, help='Output file to save instructions')
    parser.add_argument('--difficulty',         type=str, default="EASY", help='EASY/MEDIUM/HARD')


    args = parser.parse_args()

    # Fetch API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key.strip())

    main(args.count_goals, args.instructions_class, args.difficulty, args.output_file, client)
