# Verb Finder
This repository contains simple python code to extract sentences and corresponding verbs from 
pdf documents excluding the tables and figures. Specifically, the default parameters are chosen
to extract sentences and verbs **from section 6.3 of the following document: 
https://reference.opcfoundation.org/src/v104/PackML/v100/docs/readme.htm**
(somehow the link is broken, so I have put a local version of the html file: `OPC Unified Architecture.html`)

### PDF Conversion
The html is manually converted to pdf with **[this free service](https://www.hiqpdf.com/demo/HtmlFittingAndScalingOptions.aspx)** and saved as `test_pdf.pdf`. This pdf is input to our Verb Finder script. 
Following options are used for html to pdf conversion:
- **PDF settings:**
  + **Page size:** A4
  + **Margins:** 30pt (all four sides)
  + **Browser Width:** 120px
  + **Timeout:** 120sec

- **HTML Scaling and Fitting options:**
    + **Fit Width:** Checked
    + **Force Fit Width:** Checked
    + **Fit Hight:** Unchecked
    + **Postcard Mode:** Unchecked

### Environment Setup
After cloning the repo please create an anaconda environment running the following command from the project directory: <br>
`conda env create -f environment.yaml ` <br>
Then activate the environment with the command: <br>`conda activate verbfinderenv`
### Usage
Just for this specific task of sentence and verbs extraction from 6.3 section run the following command from the
project directory: <br>
`python verb_finder.py`
### Optional arguments
 - --cfg: to provide custom config.yaml file path
 - --pdf: to provide custom input pdf path
 - --pgs: to provide page range [start, end] of interest
 - --out: to provide custom output .xlsx file path

### Configuration
Regex patterns to filter out and some NLP pipeline configuration are defined in the ``config.yaml``. Main points to note are:
 - By default, verbs are extracted in lemmatized (root) forms. To detect the verbs as it is, set ``LEMMATIZE`` to `False`
 - By default, auxiliary verbs like 'be', 'can' are not detected. Set ``DETECT_AUX`` to `True` to detect those as well.
 - Single word sentences are discarded by default, considering those to be sentence segmentation errors. Set ``FILTER_SINGLE_WORDS`` to `False` to consider single word sentences as well.
