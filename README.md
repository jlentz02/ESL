# Statistical Learning on Financial Data

This project applies statistical learning techniques from *The Elements of Statistical Learning*
to real-world financial datasets, with an emphasis on model comparison and interpretability.

I plan to work in order of the techniques as they are introduced in the book starting with regression and its various flavors. The main data-set I will be working with is the 2005 Taiwanese credit loan default data-set freely available through Kaggle, but will include other data-sets when I feel it is relevant. My main aim with this project is to build a better fundamental understanding of these statistical modeling techniques that can't otherwise be communicated through the text itself. In particular, I am interested in comparing the intepretability and performance of these models. As the book is structured, the models become less interpretable (more non-linear), but does this translate to an increase in performance? 

As for the structure of this repo, I do not have any firm plans yet. I will probably divide the models by chapter/theme, but the length of implementing them may alter this plan. However, none of these techniques are really too long. A standard regression is a few lines (or even one line) of numpy. 

To clarify it again, the purpose of this project is for my own learning, and the purpose of the documentation is as a way to monitor my progress through the material. 

# Model Performance Summary

## Credit Default Data Set (Binary Classification)
| Model                  | Train Error | Test Error | Comments |
|------------------------|-------------|------------|----------|
| OLS Linear Regression  | 0.1998      | 0.2083     | Raw features, ESL §3.2|
| Ridge Regression       | 0.1992      | 0.2072     | lambda = 8 after testing on val set
