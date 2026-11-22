# Per-seed results of every classifier arm

One row per arm and seed (94 records of 30 arms) from
results/classifier_arms/arm_records.jsonl; the plateau rule, the fixed cut-offs and the false-alarm counts
are computed from the saved probabilities in results/classifier_arms/predicted_probabilities/. The OOF cost
is over the 60,000 training rows and the threshold is chosen on it; the test costs are on the 16,000-row
official test set. Oracle and FP at FN<=6 use the test labels and are diagnostics only. The CSV has further
columns: the OOF cost scaled to the 16,000 rows of the test set, the fold costs, the costs at the 0.5 and
10/510 cut-offs, the plateau threshold, the fewest false alarms at <= 9 missed failures and the
cross-validation minutes.

## Baselines and other learners

| Arm | Seed | OOF cost | Threshold | Test cost | FN | FP | Test cost 95 % interval | Full refit | Plateau rule | Oracle | FP at FN<=6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM (tuned), no augmentation | 11 | 33,290 | 4.61e-05 | 9,020 | 10 | 402 | [6,110, 12,290] | 11,840 | 9,020 | 7,710 | 471 |
| LightGBM (tuned), no augmentation | 29 | 36,770 | 4.71e-05 | 8,970 | 10 | 397 | [6,170, 12,280] | 11,460 | 9,320 | 8,520 | 679 |
| LightGBM (tuned), no augmentation | 47 | 39,230 | 5.93e-05 | 10,240 | 13 | 374 | [6,850, 13,990] | 9,580 | 10,400 | 8,450 | 591 |
| LightGBM (tuned), no augmentation | 71 | 31,160 | 5.28e-05 | 9,350 | 11 | 385 | [6,340, 12,710] | -- | 9,460 | 8,510 | 551 |
| LightGBM (tuned), no augmentation | 101 | 32,900 | 5.34e-05 | 10,360 | 13 | 386 | [6,918, 14,161] | -- | 9,880 | 9,370 | 751 |
| LightGBM (tuned), no augmentation, 3 folds | 11 | 36,310 | 2.74e-05 | 9,200 | 9 | 470 | [6,530, 12,410] | -- | 9,280 | 8,380 | 591 |
| LightGBM (tuned), no augmentation, 3 folds | 29 | 41,270 | 2.21e-05 | 9,480 | 9 | 498 | [6,780, 12,620] | -- | 9,170 | 7,810 | 543 |
| LightGBM (tuned), no augmentation, 3 folds | 47 | 37,580 | 4.84e-05 | 9,030 | 10 | 403 | [6,230, 12,340] | -- | 9,300 | 8,570 | 623 |
| XGBoost (fixed settings), no augmentation | 11 | 33,620 | 4.57e-03 | 8,480 | 8 | 448 | [5,940, 11,411] | -- | 8,440 | 7,250 | 455 |
| XGBoost (fixed settings), no augmentation | 29 | 36,050 | 3.92e-03 | 8,250 | 7 | 475 | [5,810, 11,150] | -- | 8,410 | 7,590 | 511 |
| XGBoost (fixed settings), no augmentation | 47 | 33,770 | 6.30e-03 | 8,380 | 9 | 388 | [5,710, 11,511] | -- | 7,580 | 6,710 | 423 |
| Random forest, no augmentation | 11 | 38,210 | 3.40e-02 | 9,760 | 9 | 526 | [7,090, 12,850] | -- | 9,690 | 9,100 | 649 |
| Random forest, no augmentation | 29 | 39,020 | 6.20e-02 | 9,770 | 12 | 377 | [6,650, 13,240] | -- | 10,200 | 9,440 | 648 |
| Random forest, no augmentation | 47 | 39,390 | 3.40e-02 | 9,270 | 8 | 527 | [6,719, 12,210] | -- | 10,330 | 9,240 | 633 |
| sklearn GBM, no augmentation | 11 | 44,930 | 3.31e-03 | 11,820 | 8 | 782 | [9,270, 14,780] | -- | 11,680 | 10,090 | 871 |
| sklearn GBM, no augmentation | 29 | 45,320 | 7.67e-03 | 10,650 | 11 | 515 | [7,650, 13,920] | -- | 10,310 | 10,090 | 831 |
| sklearn GBM, no augmentation | 47 | 44,780 | 6.26e-03 | 9,580 | 8 | 558 | [7,139, 12,570] | -- | 9,670 | 9,160 | 711 |

## RQ1: the cGAN of Ranasinghe et al. and its controls

| Arm | Seed | OOF cost | Threshold | Test cost | FN | FP | Test cost 95 % interval | Full refit | Plateau rule | Oracle | FP at FN<=6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cGAN +2,000 rows, LightGBM | 11 | 35,690 | 2.75e-05 | 9,500 | 10 | 450 | [6,670, 12,621] | 10,510 | 9,870 | 9,030 | 839 |
| cGAN +2,000 rows, LightGBM | 29 | 35,660 | 6.29e-05 | 10,090 | 13 | 359 | [6,759, 13,720] | 11,580 | 9,810 | 8,870 | 671 |
| cGAN +2,000 rows, LightGBM | 47 | 35,840 | 1.76e-05 | 9,230 | 8 | 523 | [6,720, 12,200] | 10,110 | 9,480 | 8,970 | 719 |
| cGAN +2,000 rows, LightGBM | 71 | 32,330 | 1.04e-04 | 11,150 | 16 | 315 | [7,460, 15,380] | -- | 11,150 | 8,650 | 647 |
| cGAN +2,000 rows, LightGBM | 101 | 34,490 | 3.77e-05 | 9,000 | 10 | 400 | [6,160, 12,270] | -- | 9,080 | 8,870 | 767 |
| 2,000 duplicated real failures, LightGBM | 11 | 35,570 | 2.46e-05 | 8,820 | 8 | 482 | [6,340, 11,881] | -- | 9,580 | 8,730 | 585 |
| 2,000 duplicated real failures, LightGBM | 29 | 37,100 | 2.45e-05 | 9,350 | 9 | 485 | [6,670, 12,490] | -- | 8,550 | 8,290 | 607 |
| 2,000 duplicated real failures, LightGBM | 47 | 36,650 | 4.40e-05 | 9,070 | 10 | 407 | [6,250, 12,500] | -- | 9,010 | 8,220 | 647 |
| 2,000 duplicated real failures, LightGBM | 71 | 32,900 | 6.24e-05 | 9,040 | 11 | 354 | [5,950, 12,660] | -- | 9,110 | 8,570 | 655 |
| 2,000 duplicated real failures, LightGBM | 101 | 35,210 | 4.06e-05 | 10,130 | 12 | 413 | [6,960, 13,670] | -- | 10,130 | 9,590 | 671 |
| cGAN +2,000 rows, random forest | 11 | 37,690 | 3.00e-02 | 9,720 | 8 | 572 | [7,180, 12,670] | -- | 9,720 | 9,010 | 601 |
| cGAN +2,000 rows, random forest | 29 | 39,270 | 3.80e-02 | 10,100 | 10 | 510 | [7,200, 13,260] | -- | 10,440 | 9,440 | 664 |
| cGAN +2,000 rows, random forest | 47 | 36,810 | 4.20e-02 | 9,790 | 10 | 479 | [6,880, 12,930] | -- | 9,800 | 9,240 | 666 |
| cGAN with the notebook's settings, random forest | 11 | 38,440 | 3.60e-02 | 10,130 | 10 | 513 | [7,230, 13,280] | -- | 10,340 | 9,470 | 648 |
| cGAN +2,000 rows, sklearn GBM | 11 | 43,040 | 6.55e-03 | 9,730 | 6 | 673 | [7,540, 12,410] | -- | 9,730 | 8,830 | 583 |
| cGAN +2,000 rows, sklearn GBM | 29 | 44,930 | 7.75e-03 | 11,530 | 11 | 603 | [8,440, 14,871] | -- | 11,730 | 10,490 | 759 |
| cGAN +2,000 rows, sklearn GBM | 47 | 42,740 | 6.33e-03 | 10,600 | 8 | 660 | [8,070, 13,501] | -- | 10,530 | 9,530 | 727 |

## RQ1: trees on the preprocessing pipeline of Beikmohammadi et al.

| Arm | Seed | OOF cost | Threshold | Test cost | FN | FP | Test cost 95 % interval | Full refit | Plateau rule | Oracle | FP at FN<=6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Preprocessing of Beikmohammadi et al. (SVM-SMOTE and RENN), LightGBM | 11 | 36,230 | 1.63e-04 | 8,850 | 6 | 585 | [6,630, 11,460] | -- | 8,920 | 8,670 | 567 |
| Preprocessing of Beikmohammadi et al. (SVM-SMOTE only), LightGBM | 11 | 36,380 | 4.99e-05 | 9,550 | 9 | 505 | [6,830, 12,790] | -- | 10,010 | 9,560 | 983 |
| Preprocessing of Beikmohammadi et al. (RENN only), LightGBM | 11 | 35,540 | 5.19e-05 | 9,840 | 9 | 534 | [7,070, 13,060] | -- | 9,320 | 8,830 | 583 |
| Preprocessing of Beikmohammadi et al. (imputation only), LightGBM | 11 | 37,670 | 1.62e-05 | 9,060 | 7 | 556 | [6,700, 11,850] | -- | 9,400 | 9,020 | 679 |

## RQ2: classical oversampling (+2,000 rows, LightGBM)

| Arm | Seed | OOF cost | Threshold | Test cost | FN | FP | Test cost 95 % interval | Full refit | Plateau rule | Oracle | FP at FN<=6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SMOTE, +2,000 rows | 11 | 35,270 | 1.36e-04 | 9,860 | 13 | 336 | [6,560, 13,460] | -- | 10,010 | 9,570 | 966 |
| SMOTE, +2,000 rows | 29 | 35,750 | 3.34e-05 | 11,050 | 12 | 505 | [7,980, 14,550] | -- | 10,790 | 9,730 | 951 |
| SMOTE, +2,000 rows | 47 | 38,480 | 2.80e-05 | 10,950 | 11 | 545 | [7,920, 14,520] | -- | 10,540 | 10,270 | 1,007 |
| Borderline-SMOTE, +2,000 rows | 11 | 37,520 | 1.36e-04 | 10,310 | 14 | 331 | [6,989, 13,970] | -- | 10,130 | 9,630 | 871 |
| Borderline-SMOTE, +2,000 rows | 29 | 36,560 | 2.72e-05 | 9,540 | 9 | 504 | [6,870, 12,660] | -- | 9,650 | 9,240 | 791 |
| Borderline-SMOTE, +2,000 rows | 47 | 36,770 | 2.73e-05 | 10,520 | 11 | 502 | [7,480, 14,020] | -- | 10,370 | 9,490 | 903 |
| ADASYN, +2,000 rows | 11 | 37,790 | 3.69e-05 | 10,960 | 12 | 496 | [7,880, 14,420] | -- | 10,010 | 9,730 | 895 |
| ADASYN, +2,000 rows | 29 | 39,440 | 1.26e-04 | 10,580 | 14 | 358 | [7,200, 14,350] | -- | 10,540 | 10,050 | 1,015 |
| ADASYN, +2,000 rows | 47 | 37,280 | 3.56e-05 | 10,630 | 11 | 513 | [7,630, 14,110] | -- | 10,300 | 9,600 | 1,055 |
| SVM-SMOTE, +2,000 rows | 11 | 36,680 | 3.21e-05 | 10,420 | 11 | 492 | [7,470, 13,690] | -- | 10,060 | 9,140 | 839 |
| SVM-SMOTE, +2,000 rows | 29 | 35,930 | 2.30e-05 | 10,890 | 11 | 539 | [7,880, 14,120] | -- | 10,890 | 9,200 | 655 |
| SVM-SMOTE, +2,000 rows | 47 | 35,000 | 5.20e-05 | 9,620 | 11 | 412 | [6,570, 13,100] | -- | 9,450 | 9,380 | 735 |

## RQ2: statistical and deep generators (+2,000 rows, LightGBM)

| Arm | Seed | OOF cost | Threshold | Test cost | FN | FP | Test cost 95 % interval | Full refit | Plateau rule | Oracle | FP at FN<=6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian copula, +2,000 rows | 11 | 35,600 | 4.59e-05 | 9,280 | 10 | 428 | [6,450, 12,410] | -- | 9,280 | 9,000 | 679 |
| Gaussian copula, +2,000 rows | 29 | 34,070 | 4.61e-05 | 9,400 | 10 | 440 | [6,580, 12,510] | -- | 9,820 | 8,870 | 703 |
| Gaussian copula, +2,000 rows | 47 | 37,250 | 2.10e-05 | 9,430 | 8 | 543 | [6,950, 12,360] | -- | 8,870 | 8,550 | 647 |
| TVAE, +2,000 rows | 11 | 36,020 | 4.26e-05 | 9,330 | 10 | 433 | [6,340, 12,720] | -- | 9,650 | 8,920 | 743 |
| TVAE, +2,000 rows | 29 | 33,650 | 3.61e-05 | 9,950 | 11 | 445 | [6,880, 13,391] | -- | 9,810 | 9,060 | 815 |
| TVAE, +2,000 rows | 47 | 36,920 | 1.34e-04 | 10,110 | 14 | 311 | [6,670, 13,980] | -- | 9,320 | 8,970 | 855 |
| CTGAN, +2,000 rows | 11 | 36,740 | 4.12e-05 | 10,240 | 12 | 424 | [6,990, 13,711] | -- | 10,240 | 9,680 | 711 |
| CTGAN, +2,000 rows | 29 | 35,870 | 7.65e-05 | 9,980 | 13 | 348 | [6,579, 13,660] | -- | 9,920 | 8,460 | 559 |
| CTGAN, +2,000 rows | 47 | 33,980 | 4.18e-05 | 9,860 | 11 | 436 | [6,790, 13,360] | -- | 9,450 | 9,430 | 663 |
| CTAB-GAN+, +2,000 rows | 11 | 35,780 | 8.04e-05 | 8,890 | 11 | 339 | [5,840, 12,320] | -- | 9,260 | 8,350 | 535 |
| CTAB-GAN+, +2,000 rows | 29 | 37,550 | 1.50e-05 | 8,900 | 7 | 540 | [6,560, 11,710] | -- | 8,800 | 8,200 | 551 |
| CTAB-GAN+, +2,000 rows | 47 | 37,490 | 6.06e-05 | 9,770 | 12 | 377 | [6,570, 13,370] | -- | 8,900 | 8,520 | 863 |
| TabDDPM, +2,000 rows | 11 | 35,840 | 2.05e-05 | 9,540 | 9 | 504 | [6,900, 12,550] | -- | 9,420 | 8,760 | 719 |
| TabDDPM, +2,000 rows | 29 | 35,570 | 3.05e-05 | 9,680 | 10 | 468 | [6,889, 12,790] | -- | 9,040 | 8,870 | 703 |
| TabDDPM, +2,000 rows | 47 | 35,480 | 3.48e-05 | 9,550 | 10 | 455 | [6,690, 12,780] | -- | 9,430 | 9,430 | 671 |
| Forest-Flow +2,000 rows, LightGBM | 11 | 36,800 | 2.97e-05 | 9,600 | 10 | 460 | [6,780, 12,760] | -- | 9,610 | 8,790 | 831 |
| Forest-Flow +2,000 rows, LightGBM | 29 | 35,510 | 1.72e-05 | 9,970 | 9 | 547 | [7,290, 13,150] | -- | 9,990 | 8,630 | 719 |
| Forest-Flow +2,000 rows, LightGBM | 47 | 37,400 | 2.68e-05 | 9,370 | 9 | 487 | [6,640, 12,590] | -- | 9,370 | 9,240 | 655 |
| TabEBM +2,000 rows, LightGBM (GPU) | 11 | 34,550 | 2.91e-05 | 9,700 | 10 | 470 | [6,899, 12,810] | -- | 9,740 | 9,110 | 735 |
| TabEBM +2,000 rows, LightGBM (GPU) | 29 | 35,180 | 3.34e-05 | 9,520 | 10 | 452 | [6,699, 12,630] | -- | 9,550 | 9,110 | 846 |
| TabEBM +2,000 rows, LightGBM (GPU) | 47 | 36,620 | 1.57e-05 | 10,100 | 9 | 560 | [7,460, 13,160] | -- | 9,940 | 9,240 | 727 |

## RQ2: other learners and ensembles (no synthetic rows)

| Arm | Seed | OOF cost | Threshold | Test cost | FN | FP | Test cost 95 % interval | Full refit | Plateau rule | Oracle | FP at FN<=6 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabICLv2, bagged contexts, no tuning | 11 | 32,270 | 5.12e-02 | 7,320 | 5 | 482 | [5,310, 9,730] | -- | 7,420 | 7,280 | 463 |
| TabICLv2, bagged contexts, no tuning | 29 | 31,400 | 6.86e-02 | 7,820 | 7 | 432 | [5,460, 10,721] | -- | 7,940 | 7,490 | 463 |
| TabICLv2, bagged contexts, no tuning | 47 | 31,070 | 5.80e-02 | 7,980 | 7 | 448 | [5,620, 10,880] | -- | 8,080 | 7,330 | 479 |
| TabICLv2, bagged contexts, no tuning | 71 | 32,450 | 8.69e-02 | 8,290 | 9 | 379 | [5,710, 11,460] | -- | 8,660 | 6,980 | 471 |
| TabICLv2, bagged contexts, no tuning | 101 | 29,540 | 1.11e-01 | 9,700 | 13 | 320 | [6,469, 13,420] | -- | 9,840 | 7,780 | 511 |
| LightGBM (tuned) + missingness indicators | 11 | 33,500 | 1.71e-05 | 8,960 | 7 | 546 | [6,600, 11,921] | -- | 9,270 | 8,610 | 615 |
| LightGBM (tuned) + missingness indicators | 29 | 37,220 | 2.75e-05 | 9,180 | 9 | 468 | [6,490, 12,380] | -- | 9,930 | 8,370 | 630 |
| LightGBM (tuned) + missingness indicators | 47 | 37,670 | 4.55e-05 | 10,040 | 12 | 404 | [6,910, 13,660] | -- | 9,560 | 8,320 | 551 |
| Ensemble (LightGBM + XGBoost + CatBoost), no indicators | 11 | 33,080 | 6.65e-04 | 7,520 | 5 | 502 | [5,520, 9,920] | -- | 7,440 | 7,460 | 447 |
| Ensemble (LightGBM + XGBoost + CatBoost), no indicators | 29 | 35,240 | 7.37e-04 | 8,220 | 7 | 472 | [5,900, 11,030] | -- | 8,150 | 7,460 | 519 |
| Ensemble (LightGBM + XGBoost + CatBoost), no indicators | 47 | 34,280 | 5.77e-04 | 7,860 | 5 | 536 | [5,850, 10,281] | -- | 7,590 | 7,520 | 463 |
| Ensemble (LightGBM + XGBoost + CatBoost), no indicators | 71 | 30,830 | 1.35e-03 | 9,200 | 11 | 370 | [6,210, 12,700] | -- | 9,160 | 7,390 | 439 |
| Ensemble (LightGBM + XGBoost + CatBoost), no indicators | 101 | 31,940 | 1.26e-03 | 8,820 | 10 | 382 | [5,980, 12,081] | -- | 8,790 | 7,630 | 463 |
| Ensemble (LightGBM + XGBoost + CatBoost) + missingness indicators | 11 | 31,910 | 9.85e-04 | 7,790 | 7 | 429 | [5,470, 10,530] | -- | 8,300 | 7,540 | 471 |
| Ensemble (LightGBM + XGBoost + CatBoost) + missingness indicators | 29 | 35,480 | 9.72e-04 | 8,730 | 9 | 423 | [5,960, 11,780] | -- | 8,100 | 7,730 | 479 |
| Ensemble (LightGBM + XGBoost + CatBoost) + missingness indicators | 47 | 33,200 | 7.99e-04 | 8,580 | 8 | 458 | [6,080, 11,661] | -- | 8,590 | 7,730 | 487 |
| Ensemble (LightGBM + XGBoost + CatBoost) + missingness indicators | 71 | 32,180 | 1.59e-03 | 9,920 | 13 | 342 | [6,550, 13,681] | -- | 9,870 | 7,470 | 447 |
| Ensemble (LightGBM + XGBoost + CatBoost) + missingness indicators | 101 | 32,450 | 1.25e-03 | 8,830 | 10 | 383 | [6,010, 12,120] | -- | 9,070 | 7,810 | 487 |
| Ensemble + indicators + TabICLv2 member | 11 | 31,610 | 2.70e-03 | 7,640 | 7 | 414 | [5,330, 10,490] | -- | 8,670 | 7,170 | 447 |
| Ensemble + indicators + TabICLv2 member | 29 | 34,760 | 2.58e-03 | 7,690 | 7 | 419 | [5,390, 10,550] | -- | 8,760 | 7,300 | 495 |
| Ensemble + indicators + TabICLv2 member | 47 | 31,910 | 2.77e-03 | 8,060 | 8 | 406 | [5,530, 11,090] | -- | 8,090 | 6,900 | 511 |
| Ensemble + indicators + TabICLv2 member | 71 | 30,950 | 4.15e-03 | 9,430 | 12 | 343 | [6,329, 13,071] | -- | 7,430 | 7,420 | 543 |
| Ensemble + indicators + TabICLv2 member | 101 | 31,520 | 7.55e-03 | 12,180 | 19 | 268 | [8,020, 16,590] | -- | 11,760 | 7,470 | 447 |

Columns:

- Arm: the arm's label (CSV: arm is the name in arm_records.jsonl, label and group the display names).
- Seed: random seed of the folds, the models and the generators (CSV: seed).
- OOF cost: cost of the out-of-fold predictions of the 60,000 training rows at the OOF threshold (CSV: oof_cost, with oof_fn and oof_fp).
- Threshold: the threshold that minimises the OOF cost (CSV: oof_threshold).
- Test cost, FN, FP: cost, missed failures and false alarms on the test set with the fold models' average probability at the OOF threshold (CSV: test_cost, test_fn, test_fp).
- Test cost 95 % interval: percentile bootstrap interval of the test cost with the predictions held fixed (CSV: test_cost_interval_low, test_cost_interval_high).
- Full refit: test cost of one model refitted on all training rows, at the same threshold; only for the arms and seeds where it was run (CSV: full_refit_test_cost, _fn, _fp).
- Plateau rule: test cost at the plateau threshold (CSV: test_cost_at_plateau_threshold, plateau_threshold).
- Oracle: lowest test cost over all thresholds (CSV: test_oracle_minimum_cost).
- FP at FN<=6: fewest test false alarms with at most 6 missed failures (CSV: test_min_fp_at_most_6_fn).

Unrounded values: `classifier_arms_per_seed.csv` in this folder.
