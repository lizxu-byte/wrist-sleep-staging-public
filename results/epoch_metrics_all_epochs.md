| task | model | kappa_pooled | kappa_subject_mean | kappa_subject_sd | accuracy | sens_wake | sens_sleep | sens_NREM | sens_REM | sens_light | sens_deep | wake_specificity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sleep_wake | logreg | 0.409 | 0.405 | 0.179 | 0.863 | 0.696 | 0.879 | nan | nan | nan | nan | 0.696 |
| sleep_wake | rf | 0.518 | 0.512 | 0.155 | 0.912 | 0.634 | 0.940 | nan | nan | nan | nan | 0.634 |
| wake_nrem_rem | logreg | 0.407 | 0.401 | 0.176 | 0.677 | 0.648 | nan | 0.680 | 0.677 | nan | nan | 0.648 |
| wake_nrem_rem | rf | 0.470 | 0.454 | 0.170 | 0.740 | 0.639 | nan | 0.807 | 0.574 | nan | nan | 0.639 |
| wake_light_deep_rem | logreg | 0.356 | 0.348 | 0.128 | 0.541 | 0.635 | nan | nan | 0.633 | 0.435 | 0.756 | 0.635 |
| wake_light_deep_rem | rf | 0.411 | 0.405 | 0.157 | 0.624 | 0.638 | nan | nan | 0.587 | 0.657 | 0.541 | 0.638 |
