# Baryogenesis

This repository contains all the code used to perform the calculations and generate the plots in arXiv:260X.XXXXX.  If you have any questions, please contact Saarik Kalia (skalia@ifae.es).

This repository contains the following:
- collider.py: Calculates single-dijet collider bound on M (Mmax parameter for 'uds' and 'dsc' cases in constraint.py)
  - Compares dijet production cross-section to CMS data and finds maximum M where production exceeds data
  - Returns Mmax and generates associated plot
  - Can use either 7 TeV or 13 TeV data
- constraint.py: Generates Figs. 4 and 5, using results of contour.py and collider.py
- contour.py: Generates contour*.npz files, which contain data for contours in Figs. 4 and 5
- dijet_high.txt: 13 TeV single-dijet CMS data (from arXiv:1911.03947)
- dijet_low.txt: 7 TeV single-dijet CMS data (from arXiv:1010.0203)
- evolution.py: Generates Fig. 2
- gstar.txt: Profile for relativistic SM degrees of freedom, used in cosmological evolutions (from arXiv:1609.04979)
- yield.py: Generates Fig. 3
