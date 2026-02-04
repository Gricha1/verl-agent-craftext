#!/bin/bash

# Количество целей (по умолчанию 2, можно изменить при запуске)
COUNT_GOALS=6

echo "Using COUNT_GOALS=$COUNT_GOALS"

# Функция для выполнения генерации и проверки
generate_and_validate() {
    local class=$1
    local difficulty=$2
    local output_file=$3

    echo "Generating instructions for $class with difficulty $difficulty..."
    python3 dialogue.py --count_goals "$COUNT_GOALS" \
                       --instructions_class "$class" \
                       --output_file "$output_file" \
                       --difficulty "$difficulty"

    echo "Validating generated instructions: $output_file..."
    python3 validator.py "$output_file"

    if [ $? -eq 0 ]; then
        echo "Validation passed for $output_file!"
    else
        echo "Validation failed for $output_file. Please check the output."
    fi
}

# # Генерация инструкций и проверка
# generate_and_validate "building_line" "MEDIUM" "instructions/instructions_line_medium.txt"
# generate_and_validate "building_line" "EASY" "instructions/instructions_line_easy.txt"

#generate_and_validate "building_star" "EASY"   "/home/dmitriyl/CrafText/craftext/scenarios/jax_build_star/instructions/train/easy/test.txt"
#generate_and_validate "building_star" "MEDIUM" "/home/dmitriyl/CrafText/craftext/scenarios/jax_build_star/instructions/train/medium/test.txt"

# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# generate_and_validate "time_base"           "EASY"   "/home/dmitriyl/CrafText/craftext/scenarios/jax_night_base/instructions/train/easy/train.txt"
# generate_and_validate "time_base"           "MEDIUM" "/home/dmitriyl/CrafText/craftext/scenarios/jax_night_base/instructions/train/medium/train.txt"

# generate_and_validate "time_did_not_fight"  "EASY"   "/home/dmitriyl/CrafText/craftext/scenarios/jax_night_did_not_fight/instructions/train/easy/train.txt"
# generate_and_validate "time_did_not_fight"  "MEDIUM" "/home/dmitriyl/CrafText/craftext/scenarios/jax_night_did_not_fight/instructions/train/medium/train.txt"

# generate_and_validate "time_at_shelter"     "EASY"   "/home/dmitriyl/CrafText/craftext/scenarios/jax_night_at_shelter/instructions/train/easy/train.txt"
# generate_and_validate "time_at_shelter"     "MEDIUM" "/home/dmitriyl/CrafText/craftext/scenarios/jax_night_at_shelter/instructions/train/medium/train.txt"

# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# generate_and_validate "time_base"           "EASY"   "/home/dmitriyl/CrafText/craftext/scenarios/jax_night_base/instructions/test/easy/train.txt"
# generate_and_validate "time_base"           "MEDIUM" "/home/dmitriyl/CrafText/craftext/scenarios/jax_night_base/instructions/test/medium/train.txt"

# generate_and_validate "time_did_not_fight"  "EASY"   "/home/dmitriyl/CrafText/craftext/scenarios/jax_night_did_not_fight/instructions/test/easy/train.txt"
# generate_and_validate "time_did_not_fight"  "MEDIUM" "/home/dmitriyl/CrafText/craftext/scenarios/jax_night_did_not_fight/instructions/test/medium/train.txt"

# generate_and_validate "time_at_shelter"     "EASY"   "/home/dmitriyl/CrafText/craftext/scenarios/jax_night_at_shelter/instructions/test/easy/train.txt"
# generate_and_validate "time_at_shelter"     "MEDIUM" "/home/dmitriyl/CrafText/craftext/scenarios/jax_night_at_shelter/instructions/test/medium/train.txt"

# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# generate_and_validate "time_placement"  "EASY"   "/home/dmitriyl/CrafText/craftext/scenarios/time_placement/instructions/train/easy/train.txt"
# generate_and_validate "time_placement"  "MEDIUM" "/home/dmitriyl/CrafText/craftext/scenarios/time_placement/instructions/train/medium/train.txt"


# generate_and_validate "time_placement"  "EASY"   "/home/dmitriyl/CrafText/craftext/scenarios/time_placement/instructions/test/easy/train.txt"
# generate_and_validate "time_placement"  "MEDIUM" "/home/dmitriyl/CrafText/craftext/scenarios/time_placement/instructions/test/medium/train.txt"


generate_and_validate "time_cosntrained_placment"  "EASY"   "/home/dmitriyl/CrafText/craftext/scenarios/jax_time_cosntrained_placment/instructions/train/easy/train.txt"
generate_and_validate "time_cosntrained_placment"  "MEDIUM" "/home/dmitriyl/CrafText/craftext/scenarios/jax_time_cosntrained_placment/instructions/train/medium/train.txt"


# generate_and_validate "jax_time_cosntrained_placment"  "EASY"   "/home/dmitriyl/CrafText/craftext/scenarios/time_placement/instructions/test/easy/train.txt"
# generate_and_validate "jax_time_cosntrained_placment"  "MEDIUM" "/home/dmitriyl/CrafText/craftext/scenarios/time_placement/instructions/test/medium/train.txt"


# generate_and_validate "building_square" "MEDIUM" "instructions/instructions_square_medium.txt"
# generate_and_validate "building_square" "EASY" "instructions/instructions_square_easy.txt"

# generate_and_validate "localization_placing" "MEDIUM" "instructions/instructions_localization_medium.txt"
# generate_and_validate "localization_placing" "EASY" "instructions/instructions_localization_easy.txt"

# generate_and_validate "conditonal_placing" "MEDIUM" "instructions/instructions_conditonal_placing_medium.txt"
# generate_and_validate "conditonal_placing" "EASY" "instructions/instructions_conditonal_placing_easy.txt"

echo "All instructions generated and validated successfully!"
