# 🔐 PhishOracle

Official implementation of  
**"From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks"**  
Accepted at **ACM Digital Threats: Research and Practice (DTRAP), 2025** [Paper](https://dl.acm.org/doi/10.1145/3737295)

---

## 📖 Abstract

Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.
  
  To address these challenges, we develop `PhishOracle`, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of three existing task-specific models - **Stack model**, **VisualPhishNet**, and **Phishpedia** - against `PhishOracle`-generated adversarial phishing webpages and observe a significant drop in their detection rates. In contrast, a **multimodal large language model** (MLLM)-based phishing detector demonstrates stronger robustness against these adversarial attacks but still is prone to evasion. Our findings highlight the vulnerability of phishing detection models to adversarial attacks, emphasizing the need for more robust detection approaches. Furthermore, we conduct a user study to evaluate whether `PhishOracle`-generated adversarial phishing webpages can deceive users. The results show that many of these phishing webpages evade not only existing detection models but also users.

---

## 🚀 Key Contributions

- 🔧 **PhishOracle Tool**  
   We propose `PhishOracle`, a phishing webpage *generator* capable of producing adversarial phishing webpages by randomly embedding content-based and visual-based phishing features into legitimate webpages.

- 📊 **Comprehensive Evaluation**  
  We carry out comprehensive evaluations to evaluate the robustness of ML, DL and LLM-based phishing webpage detectors using `PhishOracle`-generated pages. To the best of our knowledge, this is the first work to do so with one tool.

- 🧠 **User Study**  
  Demonstrates that **~48%** of generated phishing webpages are mistakenly classified as legitimate by real users.

- 🧪 **Real-world Validation**  
  We further validate the effectiveness of adversarial phishing webpages generated by `PhishOracle` by testing them against **90+** security vendors on **VirusTotal**.

- 📂 **Dataset Contribution**  
  Finally, we contribute a dataset ([DATASET](https://drive.google.com/drive/folders/1rvzo5EGu78RnhXzcRL8OH_28Yo_Nxj6Z?usp=sharing)) containing **~10K** legitimate webpage screenshots, on which we manually labelled the identity logos.

- ⚙️ **PhishOracle Web App**
  Avialable at [WebApp](https://github.com/LetsBeSecure/PhishOracle-Webapp)

---

## `PhishOracle`: Generating Adversarial Phishing Webpages

`PhishOracle` generates different variants of adversarial phishing webpages for a given legitimate webpage by adding randomly selected content and visual-based phishing features.

The file structure is as follows:
  - `urls.csv`: Add legitimate URLs into this file.  
  - `add_total_features.txt`: Add an integer between 4 and 10 (represents the number of phishing features to embed in a legitimate webpage).  
  - `download_legitimate_webpages.py`: This script reads the URL(s) from `urls.csv` and downloads the webpages into a directory. Please refer to the [Phish-Blitz GitHub repository](https://github.com/Duddu-Hriday/Phish-Blitz) for the latest code to download webpages.  
  - `adding_15_features.py`: Contains 15 content-based phishing features.
  - `add_visual_features_main.py`: The main script to run; it adds visual-based features (specific to logos) and calls `adding_15_features.py` to add the remaining content-based phishing features.

> **NOTE:** Please replace `"/path/to/downloaded_webpage/folder/"` with your actual folder directory path in `add_visual_features_main.py`.

Finally, running this pipeline will generate an adversarial phishing webpage!

### Usage

Run the following commands to (i) download legitimate webpages, and (ii) generate adversarial phishing webpages for the legitimate webpages:

```bash
cd PhishOracle_Tool
python download_legitimate_webpages.py
python add_visual_features_main.py
```
---

## 🧪 Experimental Setup and Modifications

In this work, we evaluate the robustness of several phishing webpage detection models when exposed to adversarially generated phishing webpages.

### ✅ Models Used

We cloned the following models from the official [Phishpedia GitHub repository](https://github.com/lindsey98/Phishpedia):

- **Stack model**
- **Phishpedia**
- **VisualPhishNet**

Each of these models was **retrained** on our **latest collected dataset** of phishing and legitimate webpages and subsequently **evaluated against adversarial phishing webpages** generated using `PhishOracle`.

---

### 🧩 Enhancements to VisualPhishNet

In this experiment, we use phishing and legitimate webpage screenshot datasets. The dataset consists of ~900 phishing samples targeting 41 brands, along with ~600 legitimate webpage screenshots for these 41 target brands, collected by visiting their official website hyperlinks. Additionally, to ensure an unbiased training process, we include ~300 legitimate webpage screenshots from non-target brands. This dataset is used to train the triplet CNN model in VisualPhishNet for visual similarity-based phishing detection. We follow the 60:40 train-test split to evaluate the model’s performance effectively.

> 📂 **Dataset available here**: [DATASET](https://drive.google.com/drive/folders/1-uFoOrVRQehAgRy-M6lGicgZnNMo2se1?usp=sharing)

### 🧩 Enhancements to Phishpedia

To enhance Phishpedia's logo detection component, we manually annotated logos in **9067 legitimate webpage screenshots**. We used [MakeSense.ai](https://makesense.ai/) to draw bounding boxes around the logos, exporting the annotations in **XML format**. These were then converted to **COCO format**, which was used to retrain the object detection model within Phishpedia.

> 📂 **Annotated dataset available here**: [DATASET](https://drive.google.com/drive/folders/1rvzo5EGu78RnhXzcRL8OH_28Yo_Nxj6Z?usp=sharing)

---

### 🔍 LLM-Based Phishing Detection (LLM-PD)

To ensure a fair comparison with existing phishing detection models, we utilize two Gemini-based variants: LLM-PD<sup>H</sup> and LLM-PD<sup>S</sup>. The Stack model, which relies on URL and HTML-based features, is evaluated using a set of adversarial phishing webpages that target its specific detection modality. Accordingly, we use the HTML prompt  from [GitHub_Repository](https://github.com/jehleekr/multimodal_llm_phishing_detection) repository and provide the corresponding HTML contents as input to the LLM-based pipeline, referring to this variant as LLM-PD<sup>H</sup>.

Similarly, Phishpedia, which identifies brands in webpage screenshots, is evaluated using adversarial examples designed to challenge visual brand recognition. We therefore use the screenshot prompt from the same repository and input the corresponding webpage screenshots into the LLM-based pipeline, referring to this as LLM-PD<sup>S</sup>.

This approach ensures that the performance of the LLM-based approach is directly compared to state-of-the-art phishing webpage detection models under the same attack conditions tailored to each model’s input modality.

---

## 📄 Citation

If you find our work valuable or insightful for your research, we would greatly appreciate it if you consider citing our paper:
- 📄 [arXiv preprint](https://arxiv.org/pdf/2407.20361)

```bibtex
@article{kulkarni2024ml,
  title={From ml to llm: Evaluating the robustness of phishing webpage detection models against adversarial attacks},
  author={Kulkarni, Aditya and Balachandran, Vivek and Divakaran, Dinil Mon and Das, Tamal},
  journal={arXiv preprint arXiv:2407.20361},
  year={2024}
}
