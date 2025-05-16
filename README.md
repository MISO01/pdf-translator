# pdf-translator
Type-aware PDF translator: Upload English/Chinese documents, auto-detect type (e.g. paper/news/report), and get Korean translations with aligned view.

⸻


# DeepSeek PDF Translator

A multilingual PDF translation platform powered by the **DeepSeek API**, designed to preserve document layout and adapt translation style based on document type (e.g., research papers, news, reports).  
Built with **Streamlit**, easily deployable to Render.com or Hugging Face Spaces.

---

## 📁 Project Structure

.
├── prompts/
│   └── prompts.json       # Translation prompts by document type
├── src/
│   └── app.py             # Main Streamlit application
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
└── README.md              # This file

---

## ✅ Features

- Upload English PDF documents and translate to **Chinese**
- Detects document type and applies context-aware translation prompts
- Preserves layout with **side-by-side** original & translated text
- Sentence-level **scroll and highlight sync**
- Future support for PDF export and citation-aware answers

---

## ⚙️ Local Setup

### 1. Clone this repo

```bash
git clone https://github.com/your-username/document-translate-miso.git
cd document-translate-miso

2. Install dependencies

pip install -r requirements.txt

3. Set up environment variable

You can set your DeepSeek API Key via shell:

export DEEPSEEK_API_KEY=your_deepseek_api_key

Or use a .env file with python-dotenv support.

4. Run the app

streamlit run src/app.py


⸻

🌍 Deployment on Render.com
	1.	Push this project to GitHub
	2.	Go to Render.com, create a New Web Service
	3.	Set the following configs:

	•	Build Command:
pip install -r requirements.txt
	•	Start Command:
streamlit run src/app.py --server.port $PORT --server.headless true
	•	Environment Variables:
Add DEEPSEEK_API_KEY with your actual key

	4.	After deployment, you’ll get a public .onrender.com link.

⸻

📸 Screenshots (Optional)

You can add UI screenshots or GIFs here to show the translation in action.

⸻

📜 License

This project is released under the MIT License.

⸻

👩‍💻 Author

Created by Miso Kim
Feel free to open issues or pull requests!

---
