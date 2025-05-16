import os
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"  # Streamlit 사용 통계 수집 비활성화
import streamlit as st
import fitz  # PyMuPDF
from langdetect import detect, LangDetectException
import re # For keyword matching
import json # For loading prompts
import requests # For DeepSeek API call
import nltk

st.set_page_config(layout="wide")

st.title("PDF 문서 번역 플랫폼 (DeepSeek API)")

# --- NLTK Setup ---
try:
    nltk.data.find("tokenizers/punkt")
except LangDetectException:
    nltk.download("punkt", quiet=True)

# --- Constants and Configuration ---
UPLOAD_DIR = "./uploads"
PROMPT_FILE = "./prompts/prompts.json"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

DOC_TYPE_KEYWORDS = {
    "논문": ["abstract", "introduction", "methodology", "results", "discussion", "conclusion", "references", "doi", "journal", "conference", "摘要", "引言", "方法", "结果", "讨论", "结论", "参考文献"],
    "신문": ["breaking news", "report", "source", "updated", "published", "headline", "article", "correspondent", "新闻", "报道", "来源", "更新", "发布", "头条", "文章", "记者"],
    "보고서": ["report", "study", "analysis", "findings", "executive summary", "recommendations", "appendix", "报告", "研究", "分析", "发现", "执行摘要", "建议", "附录"],
}

# --- Helper Functions ---
@st.cache_data
def load_prompts(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except Exception as e:
        return None, f"프롬프트 로드 에러: {str(e)}"

@st.cache_data
def extract_text_from_pdf(pdf_file_path):
    try:
        doc = fitz.open(pdf_file_path)
        text = "".join([page.get_text() for page in doc])
        doc.close()
        return (text, None) if text.strip() else (None, "PDF에서 텍스트를 추출할 수 없습니다. 이미지 PDF이거나 빈 문서일 수 있습니다.")
    except Exception as e:
        return None, f"PDF 텍스트 추출 에러: {str(e)}"

@st.cache_data
def detect_language_cached(text):
    try:
        lang = detect(text)
        return ('zh', None) if lang.startswith('zh') else (lang, None)
    except Exception as e:
        return None, f"언어 감지 에러: {str(e)}"

@st.cache_data
def detect_document_type_cached(text, lang):
    if not text: return "알 수 없음", "텍스트 내용이 비어 있습니다."
    text_lower = text.lower()
    scores = {doc_type: 0 for doc_type in DOC_TYPE_KEYWORDS}
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if lang == 'zh':
                scores[doc_type] += text_lower.count(keyword)
            else:
                scores[doc_type] += len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))
    if any(scores.values()):
        detected_type = max(scores, key=scores.get)
        return (detected_type, None) if scores[detected_type] >= 2 else ("통용", None)
    return "통용", "특정 문서 유형을 식별할 수 없습니다. 일반 번역 스타일을 사용합니다."

def get_translation_prompt(doc_type, source_lang, all_prompts):
    prompt_key = f"{doc_type}_{source_lang}" if source_lang == 'zh' else doc_type
    selected_prompt_set = all_prompts.get(prompt_key)
    if not selected_prompt_set:
        fallback_key = "통용_zh" if source_lang == 'zh' else "통용"
        selected_prompt_set = all_prompts.get(fallback_key)
        return selected_prompt_set, f"{doc_type} ({source_lang}) 타입의 특정 프롬프트를 찾을 수 없습니다. 일반 프롬프트로 대체합니다."
    return selected_prompt_set, None

def translate_text_deepseek(text_to_translate, system_prompt, user_prompt_template, source_lang_full):
    if not DEEPSEEK_API_KEY: return None, "DeepSeek API Key가 설정되지 않았습니다."
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    user_prompt = user_prompt_template.format(source_lang_full=source_lang_full, text=text_to_translate)
    # 토큰 수 예상 (문자 / 3으로 단순 추정)
    estimated_tokens = len(system_prompt) / 3 + len(user_prompt) / 3
    max_output_tokens = 4000 # 기본값
    if estimated_tokens > 28000: # 32k 모델 가정, 출력을 위한 공간 확보
        return None, "입력 텍스트가 너무 깁니다. API 처리 한도를 초과했습니다. 더 짧은 문서를 시도하세요."
    elif estimated_tokens > 12000: # 16k 모델 가정
        max_output_tokens = 2000
    
    # 긴 텍스트의 경우 청킹이 필요합니다. 이 구현은 단순화된 단일 호출입니다.
    # 현재는 전체 텍스트를 한 번에 보냅니다.

    payload = {
        "model": "deepseek-chat", 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": max_output_tokens 
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        if result.get("choices") and result["choices"][0].get("message") and result["choices"][0]["message"].get("content"):
            return result["choices"][0]["message"]["content"], None
        return None, f"API 응답 형식이 예상과 다릅니다: {result.get('error', result)}"
    except requests.exceptions.Timeout:
        return None, "DeepSeek API 요청 시간 초과. 나중에 다시 시도하거나 더 짧은 텍스트를 사용하세요."
    except requests.exceptions.RequestException as e:
        return None, f"DeepSeek API 요청 실패: {str(e)}"
    except Exception as e:
        return None, f"번역 중 알 수 없는 오류 발생: {str(e)}"

@st.cache_data
def segment_into_sentences_cached(text, lang='english'):
    # NLTK의 문장 토크나이저는 언어에 따라 다릅니다.
    # 중국어의 경우 NLTK의 기본값이 최적이 아닐 수 있어 특정 중국어 문장 토크나이저를 사용하는 것이 좋습니다.
    # 그러나 punkt는 여러 경우에 괜찮게 작동합니다.
    nltk_lang = 'english' # 기본값
    if lang == 'zh':
        # NLTK가 중국어에 특정 모델을 가지고 있는지 확인하고, 없으면 일반적인 접근이나 영어 모델 사용
        # 간단함을 위해 punkt가 어느 정도 견고하기 때문에 영어를 사용합니다.
        # 현재로서는 nltk.sent_tokenize를 사용하고 결과를 관찰합니다.
        pass
    
    try:
        return nltk.sent_tokenize(text, language=nltk_lang)
    except Exception as e:
        st.warning(f"NLTK 문장 분할 오류 ({e}), 대신 줄바꿈으로 분할합니다.")
        return [s.strip() for s in text.splitlines() if s.strip()] # 대안

# --- Main Application Logic ---
all_prompts, prompt_load_error = load_prompts(PROMPT_FILE)
if prompt_load_error: st.error(f"프롬프트 설정을 로드할 수 없습니다: {prompt_load_error}"); st.stop()
if not DEEPSEEK_API_KEY: st.error("DEEPSEEK_API_KEY가 환경에 설정되지 않았습니다. 번역 기능을 사용할 수 없습니다.")

# 세션 상태 변수 초기화
if 'translated_sentences' not in st.session_state: st.session_state.translated_sentences = []
if 'original_sentences' not in st.session_state: st.session_state.original_sentences = []
if 'selected_sentence_index' not in st.session_state: st.session_state.selected_sentence_index = -1
if 'error_message' not in st.session_state: st.session_state.error_message = None
if 'doc_info' not in st.session_state: st.session_state.doc_info = {}

uploaded_file = st.file_uploader("영어 또는 중국어 PDF 파일을 업로드하세요", type="pdf", key="pdf_uploader")

if uploaded_file is not None:
    # 새 업로드 시 상태 초기화
    st.session_state.translated_sentences = []
    st.session_state.original_sentences = []
    st.session_state.selected_sentence_index = -1
    st.session_state.error_message = None
    st.session_state.doc_info = {}

    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
    st.info(f"파일 \'{uploaded_file.name}\' 업로드 성공. 처리 중...")

    with st.spinner("텍스트 추출 및 문서 분석 중..."):
        extracted_text, error = extract_text_from_pdf(file_path)
        st.session_state.error_message = error

    if not st.session_state.error_message and extracted_text:
        st.session_state.original_text_full = extracted_text # 전체 원본 텍스트 저장
        source_lang, lang_error = detect_language_cached(extracted_text)
        st.session_state.error_message = lang_error
        st.session_state.doc_info['source_lang'] = source_lang

        if not st.session_state.error_message and source_lang:
            if source_lang not in ['en', 'zh']:
                st.session_state.error_message = "현재 중국어와 영어 PDF 번역만 지원합니다."
            else:
                doc_type, type_error = detect_document_type_cached(extracted_text, source_lang)
                # detect_document_type의 type_error는 '통용'으로 되돌아가는 경우 알림에 가깝습니다
                st.session_state.doc_info['doc_type'] = doc_type
                st.session_state.doc_info['type_notice'] = type_error if doc_type != "통용" else None
                
                prompt_set, prompt_error = get_translation_prompt(doc_type, source_lang, all_prompts)
                st.session_state.error_message = prompt_error if not prompt_set else None
                st.session_state.doc_info['prompt_set'] = prompt_set
                st.session_state.doc_info['prompt_notice'] = prompt_error if prompt_set and prompt_error else None

# 전처리 중 오류가 있는 경우 표시
if st.session_state.error_message:
    st.error(st.session_state.error_message)

# 전처리가 성공한 경우 문서 정보 및 번역 버튼 표시
if 'source_lang' in st.session_state.doc_info and st.session_state.doc_info['source_lang'] and not st.session_state.error_message:
    st.write(f"감지된 문서 언어: {st.session_state.doc_info['source_lang']}")
    st.write(f"자동 감지된 문서 유형: {st.session_state.doc_info['doc_type']}")
    if st.session_state.doc_info.get('type_notice'): st.caption(st.session_state.doc_info['type_notice'])
    
    prompt_set = st.session_state.doc_info.get('prompt_set')
    if prompt_set:
        st.write("선택된 번역 프롬프트 스타일:")
        st.caption(f"System Prompt (일부): {prompt_set.get('system_prompt')[:150]}...")
        if st.session_state.doc_info.get('prompt_notice'): st.caption(st.session_state.doc_info['prompt_notice'])

        if st.button("번역 시작", key="translate_button"):
            if not DEEPSEEK_API_KEY: 
                st.error("번역 불가: DEEPSEEK_API_KEY가 설정되지 않았습니다.")
            else:
                with st.spinner(f"DeepSeek API를 사용하여 번역 중 ({st.session_state.doc_info['doc_type']}스타일)..."):
                    source_lang_full = "English" if st.session_state.doc_info['source_lang'] == 'en' else "Chinese"
                    
                    # 전체 텍스트 번역 (대용량 문서의 청킹은 향후 과제)
                    translated_content, trans_error = translate_text_deepseek(
                        st.session_state.original_text_full, 
                        prompt_set.get('system_prompt'), 
                        prompt_set.get('user_prompt_template'),
                        source_lang_full
                    )
                    if trans_error:
                        st.error(f"번역 실패: {trans_error}")
                        st.session_state.translated_sentences = []
                    elif translated_content:
                        st.success("번역 완료!")
                        # 원본 및 번역 텍스트 문장 분할
                        st.session_state.original_sentences = segment_into_sentences_cached(st.session_state.original_text_full, st.session_state.doc_info['source_lang'])
                        st.session_state.translated_sentences = segment_into_sentences_cached(translated_content, 'korean') # 대상이 한국어라고 가정
                        st.session_state.selected_sentence_index = -1 # 선택 초기화
                    else:
                        st.error("번역 실패, 유효한 결과를 받지 못했습니다.")
                        st.session_state.translated_sentences = []
    else:
        st.error("유효한 번역 프롬프트를 가져올 수 없습니다. 번역을 진행할 수 없습니다.")

# --- 하이라이팅을 통한 원본 및 번역 문장 표시 ---
if st.session_state.original_sentences and st.session_state.translated_sentences:
    st.subheader("번역 결과 (원문 vs 한국어 번역) - 문장을 클릭하여 하이라이트")
    st.caption("참고: 문장 분할이 완벽하지 않을 수 있으며, 원문과 번역문의 문장 수가 정확히 일치하지 않을 수 있습니다. 스크롤 동기화 기능은 추후 구현 예정입니다.")

    col1, col2 = st.columns(2)
    max_sentences = max(len(st.session_state.original_sentences), len(st.session_state.translated_sentences))

    # 하이라이트 및 스크롤 가능한 div를 위한 사용자 정의 CSS
    st.markdown("""
    <style>
    .sentence-box {
        height: 600px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
    .sentence {
        padding: 5px;
        margin-bottom: 5px;
        border-radius: 3px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .sentence:hover {
        background-color: #e6f7ff;
    }
    .highlighted-sentence {
        background-color: #cceeff !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # 이 버튼별 문장 접근 방식은 매우 기본적이며 긴 문서의 경우 느릴 수 있습니다.
    # JS 솔루션이 훨씬 더 부드러울 것입니다.
    with col1:
        st.markdown("**원문**")
        st.markdown("<div class='sentence-box'>", unsafe_allow_html=True)
        for i, sentence in enumerate(st.session_state.original_sentences):
            css_class = "highlighted-sentence" if i == st.session_state.selected_sentence_index else ""
            if st.button(sentence, key=f"orig_sent_{i}", help=f"문장 {i+1} 하이라이트 (원문)"):
                st.session_state.selected_sentence_index = i
                st.rerun()
            elif i == st.session_state.selected_sentence_index: # 이 실행에서 정확히 클릭되지 않은 경우에도 하이라이트된 항목이 스타일링되도록 함
                 st.markdown(f'<p class="sentence {css_class}" style="background-color: #cceeff; font-weight: bold; padding:5px; border-radius:3px; margin-bottom:5px;">{sentence}</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("**한국어 번역**")
        st.markdown("<div class='sentence-box'>", unsafe_allow_html=True)
        for i, sentence in enumerate(st.session_state.translated_sentences):
            css_class = "highlighted-sentence" if i == st.session_state.selected_sentence_index else ""
            # 길이가 다른 경우 원본 문장 인덱스에 연결 시도, 기본 접근 방식
            display_index = i
            if i < len(st.session_state.original_sentences):
                display_index = i # 하이라이트를 위해 1대1 매핑을 가정
            
            if st.button(sentence, key=f"trans_sent_{i}", help=f"문장 {i+1} 하이라이트 (번역)"):
                # 번역 문장 수가 다른 경우 여기를 클릭하면 완벽하게 매핑되지 않을 수 있습니다.
                # 현재로서는 가능한 경우 동일한 인덱스에 매핑합니다.
                st.session_state.selected_sentence_index = display_index 
                st.rerun()
            elif display_index == st.session_state.selected_sentence_index and i < len(st.session_state.translated_sentences): # 하이라이트된 항목이 스타일링되었는지 확인
                 st.markdown(f'<p class="sentence {css_class}" style="background-color: #cceeff; font-weight: bold; padding:5px; border-radius:3px; margin-bottom:5px;">{st.session_state.translated_sentences[i]}</p>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif uploaded_file is None and not st.session_state.error_message and not (st.session_state.original_sentences or st.session_state.translated_sentences) :
    st.info("PDF 파일을 업로드하여 처리를 시작하세요.")

# 오래된 파일 정리 (선택 사항, 간단한 구현)
# 프로덕션 앱의 경우 더 견고한 정리 전략을 고려하세요
# try:
#     for f_name in os.listdir(UPLOAD_DIR):
#         if (time.time() - os.path.getmtime(os.path.join(UPLOAD_DIR, f_name))) > 3600: # 1시간 이상 지난 파일
#             os.remove(os.path.join(UPLOAD_DIR, f_name))
# except Exception as e:
#     pass # 정리 오류 무시
