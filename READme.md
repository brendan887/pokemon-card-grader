# CSE 151A Group Project

## Milestone 3 - Preprocessing

Image processing, model training, and evaluation can be found in [model.ipynb](https://github.com/brendan887/pokemon-card-grader/blob/main/model.ipynb).

In this milestone, we completed data scraping as proposed in MS2. As we're working with image data, it is necessary to standardize the image size.

### Image Processing

![Image Widths](readme_images/image_width.png)
![Image Heights](readme_images/image_height.png)

The most common dimensions are ~600x800 at a 0.75:1 aspect ratio. We decided to fit the images into size 384x512 which maintains this aspect ratio and improves efficiency of training. This was achieved by scaling all images to fit within the dimensions while preserving their original aspect ratios, and then filling the background with black.

Some simple data augmentation was also applied to the images, including small shifts, rotations, and flips. This is beneficial for CV models to improve generalizability.

![Data Augmentation Results](readme_images/post_data_aug.png)

### Training and Validation

ResNet50 was used for transfer learning.

![Model Accuracy](readme_images/model_accuracy.png)
![Model Loss](readme_images/model_loss.png)

The best validation accuracy achieved was 0.3093 with a corresponding train accuracy of 0.3876.

### Conclusion

The model accuracy indicates that it is only slightly better than random selection (there are 4 classes; random would be ~0.25). However, rather than improving the classification model itself, we propose further processing the dataset. Given the task of classification into different grades, the only pertinent information in a given image is the card itself. Looking at the preview of the data augmentation, we can see some issues:

- Some images have more than one card included
- Cards can be highly skewed
- Size of cards in images may vary
- Some images include irrelevant information (i.e. background)
- PSA label contains information of the card class, which should not be given to the model

As the task at hand relies on analyzing the card itself in great detail, we need to be able to eliminate the above factors. A solution is a segmentation model to crop out and flatten cards from the image, and identify and discard images with multiple cards/no cards/no unobstructed cards.

As for the model itself, more experimentation on architectures can be conducted, including training our own model from scratch.

## Milestone 2 - Data Exploration and Initial Processing

Since we are scraping our own data, we can design our own dataset as a preprocessing step. This way, we ensure that the dataset is representative of cards that we are interested in and has a sufficient representation of cards with different features. These features include but are not limited to:

- Card type (Pokemon vs. trainer)
- Art size (half art vs. full art)
- Special variant (EX, GX, VMAX, etc.)

We have decided to scrape card images for **the top 10 most valuable cards for each set** as of 11/3/24 (with the exception of few cards that have a completely different appearance to standard cards), with a total of 978 different cards. The complete card list can be found in [here](https://docs.google.com/document/d/1S45M2bVT3rBX15cnimlXmRDwiepWfXl3hs9HDs8nDyU/edit?usp=sharing). Significant effort was required to generate this list, as well as to create a cleaned version available in `config.py`. This query provides a good distribution of cards as it automatically includes cards for each generation and the special variants for each of these generations.

For each card, we have attempted to collect 2 images for each of the following classes, though not all grades were found for each card.

- PSA 10
- PSA 9
- PSA 8
- PSA 7

This ensures that the distribution of data between classes is close to even.

### Feature Distribution

| Feature               | Count |
| --------------------- | ----- |
| Pokemon               | 789   |
| Trainer               | 189   |
| Full Art              | 187   |
| Non-Full Art\*        | 791   |
| GX                    | 72    |
| EX\*\*                | 216   |
| V                     | 42    |
| VMAX                  | 45    |
| Total Different Cards | 978   |

\*Some cards can have art spanning the card, but are not offically "Full Art" cards

\*\*There are multiple generations of different types of EX cards, so this number is higher. It is difficult to check the exact distribution between these generations without manually verifying each card.

### Class Distribution (Collection in progress)

| Class          | Count |
| -------------- | ----- |
| PSA 10         | 214   |
| PSA 9          | 280   |
| PSA 8          | 256   |
| PSA 7          | 172   |
| Total Examples | 992   |

Due to API rate limits and time required to scrape all the image data, data is still being collected as of writing. We expect >8000 examples upon completion. Given the data collected so far, the distribution between classes is quite even. Though unlikely, should the distribution be significantly skewed, we can only consider the minimum count across all classes to enforce an equal distribution.

As we are handling image data, each image has pixel values ranging from 0-255. Min-max normalization is used to scale values to 0-1.

A portion of the dataset can be found [here](https://drive.google.com/drive/folders/1NuAWu848ER3avx0a-dhP88j1xvQZjNTi?usp=sharing). This does not include the entire dataset given its size.

## Milestone 1 - Abstract

When a collector wants to validate the authenticity and quality of a trading card, they send in their cards to an authenticating company who will give the card a grade, usually from 1-10. The objective of this project is to develop a machine learning model that accurately predicts the grade/quality of a Pokémon trading card based on its image. The approach involves collecting a comprehensive dataset of images of graded cards, each annotated with its official grade. Data preprocessing techniques such as cropping to isolate the card and perspective transformation to correct for angular distortions will be used to improve the quality of the data. One solution is to train a supervised CNN to learn the visual features associated with different grading levels, such as centering, whitening, and scratching. This model aims to output a confidence score for the card's grade when presented with a new image. Another solution that will be explored is performing feature extraction and distance calculation from a perfect example of the card. These models can help evaluate the efficacy of either solution and advise if a card is worth grading.
