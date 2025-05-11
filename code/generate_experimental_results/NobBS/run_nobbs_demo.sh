#!/bin/bash

states=(
#al ak az ar ca co ct de fl ga hi id \
#il in ia ks ky la me md ma mi mn ms \
#mo mt ne nv nh nj nm ny nc nd oh ok \
#or pa ri sc sd tn tx ut vt va wa wv \
wi wy
)

for state in "${states[@]}"; do
  echo "Running NobBS for state: $state"
  Rscript NobBS_exp.R "$state" chng_count_config.R
done
