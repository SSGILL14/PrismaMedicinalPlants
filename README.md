<p align="center">
  <img src="Graphical-Abstract.jpeg" alt="Graphical abstract of the systematic review of deep learning for plant disease detection" width="900">
</p>

# Deep Learning Approaches for Plant Disease Detection in Precision Agriculture: A Systematic Literature Review

Code to reproduce Figures 2–10 and 12 from the 83-study review.

Use Python 3.12. Save `Final-Data.xlsx` from the article supplement in [`Data/`](Data/), alongside the included CSVs.

```sh
python -m pip install -r requirements.txt
python render_figures.py
```

The renderer uses Times New Roman and a Display P3 ICC profile. For custom paths, run `python render_figures.py --help`.
