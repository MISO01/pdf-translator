⸻


# pdf-translator
**Type-aware PDF translator** for Korean, Chinese, and English.  
Upload a document (PDF), auto-detect language and document type (e.g. paper, news, report), and get **structurally aligned translations** with sentence-level interactivity.

---

# DeepSeek PDF Translator

A multilingual, document-structured PDF translation platform powered by the **DeepSeek API** and **OpenAI GPT**.  
It preserves original layout, detects document type, and enables sentence-level interaction with smart re-translation.  
Built with **Streamlit**, easy to deploy on Hugging Face Spaces or Render.

---

## 📁 Project Structure

.
├── prompts/
│   └── prompts.json          # Type + language-based prompt templates
├── translation_core/
│   ├── extract_text_blocks.py
│   ├── detect_document_type.py
│   ├── load_prompts.py
│   ├── translate_text_multi_model.py
│   └── get_translation_prompt.py
├── document_exporter.py      # Word/PDF export functionality
├── src/
│   └── app.py                # Main Streamlit app
├── .env                      # (Optional) API key config
├── requirements.txt
└── README.md

---

## ✅ Features

- Upload PDF (Korean / Chinese / English supported)
- **Auto-detect document type**: paper / news / report / general
- Select source and target languages (e.g. Chinese → Korean)
- Apply **custom prompts** per document type and language pair
- Translate paragraph by paragraph using **DeepSeek** or **OpenAI GPT**
- Dual-pane view with **aligned original vs translated text**
- **Sentence-level highlight + re-translate / explain button**
- Save final translation as **.docx**
- Designed for academic, business, and policy documents

---

## ⚙️ Local Setup

### 1. Clone this repo

```bash
git clone https://github.com/your-username/pdf-translator-miso.git
cd pdf-translator-miso

2. Install dependencies

pip install -r requirements.txt

3. Set environment variables

You can either:
	•	Create a .env file with:

DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key

	•	Or export manually:

export DEEPSEEK_API_KEY=your_key
export OPENAI_API_KEY=your_key

4. Run the app

streamlit run src/app.py


⸻

🚀 Deploy on Render.com
	1.	Push this project to GitHub
	2.	Go to Render, create a New Web Service
	3.	Set the following:

	•	Build Command:

pip install -r requirements.txt


	•	Start Command:

streamlit run src/app.py --server.port $PORT --server.headless true


	•	Environment Variables:
	•	DEEPSEEK_API_KEY
	•	OPENAI_API_KEY (optional)

	4.	You’ll get a public .onrender.com link.

⸻

📸 Screenshots (Optional)

You can add a screenshot here:
	•	Original vs Translated aligned view
	•	Sentence hover & retranslate button
	•	DOCX export result

⸻

📜 License

MIT License. Feel free to fork, remix, and contribute.

⸻

👩‍💻 Author

Built by Miso Kim
If you like it or want to collaborate, feel free to open an issue or PR!

---
