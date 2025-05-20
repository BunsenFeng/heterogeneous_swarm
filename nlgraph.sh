#!/bin/bash

inertia_list=(0.1 0.2 0.3)
cognitive_coeff_list=(0.1 0.2 0.3 0.4 0.5)
social_coeff_list=(0.2 0.3 0.4 0.5 0.6)
repel_coeff_list=(0.01 0.05 0.1)
step_length_list=(0.5 0.6 0.7 0.8 0.9 1.0)

while true
do
    inertia=${inertia_list[$RANDOM % ${#inertia_list[@]} ]}
    cognitive_coeff=${cognitive_coeff_list[$RANDOM % ${#cognitive_coeff_list[@]} ]}
    social_coeff=${social_coeff_list[$RANDOM % ${#social_coeff_list[@]} ]}
    repel_coeff=${repel_coeff_list[$RANDOM % ${#repel_coeff_list[@]} ]}
    step_length=${step_length_list[$RANDOM % ${#step_length_list[@]} ]}

    python search.py \
        -n nlgraph_{$inertia}_{$cognitive_coeff}_{$social_coeff}_{$repel_coeff}_{$step_length} \
        -e exact_match \
        -d nlgraph \
        -g 0,1,2,3,4 \
        --patience 6 \
        --inertia $inertia \
        --cognitive_coeff $cognitive_coeff \
        --social_coeff $social_coeff \
        --repel_coeff $repel_coeff \
        --step_length $step_length \
        --starting_test_set_eval 0 \
        --fast_merge 1 \
        --project_name_wb hetero_nlgraph \
        --weight_randomness 1 \
        --starting_velocity_mode random \
        --repel_term 1 \
        --step_length_factor 0.95 \
        --restart_stray_particles 1 \
        --restart_patience 0.67
done