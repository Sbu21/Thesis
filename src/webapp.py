import streamlit as st
import requests
import json
import time

# --- Configuration for Flask API ---
FLASK_API_BASE_URL = "http://localhost:5001/api"

st.set_page_config(page_title="Cod Rutier Inteligent", layout="wide")
st.title("Asistent Inteligent - Codul Rutier")
st.markdown("---")


# --- Helper function to make API GET requests ---
def make_api_request(endpoint, params=None):
    """Makes a GET request to the Flask API and returns JSON response."""
    try:
        url = f"{FLASK_API_BASE_URL}/{endpoint}"
        headers = {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Eroare de conexiune: Verificati daca serverul API Flask ruleaza la {FLASK_API_BASE_URL} si este accesibil.")
    except requests.exceptions.Timeout:
        st.error("Eroare: Timeout la cererea catre API.")
    except requests.exceptions.HTTPError as e:
        error_details = str(e)
        try:
            error_json = e.response.json()
            if "error" in error_json: error_details = error_json["error"]
            if "details" in error_json: error_details += f" (Detalii: {error_json['details']})"
        except json.JSONDecodeError:
            error_details = e.response.text
        st.error(f"Eroare HTTP de la API: {e.response.status_code} - {error_details}")
    except requests.exceptions.RequestException as e:
        st.error(f"A aparut o eroare la cererea catre API: {e}")
    except json.JSONDecodeError:
        st.error("Eroare: Raspunsul de la API (chiar daca a avut succes HTTP) nu este un JSON valid.")
    return None


# --- Section 1: Search by Article Number ---
st.header("Cautare dupa Articol si Paragraf")

@st.cache_data
def get_articles_for_dropdown_cached():
    data = make_api_request("articles")
    return [""] + (data.get("articles", []) if data and isinstance(data.get("articles"), list) else [])

all_article_headers_with_blank = get_articles_for_dropdown_cached()

if not all_article_headers_with_blank or len(all_article_headers_with_blank) <= 1:
    st.warning("Nu s-au putut incarca articolele. Verificati conexiunea la API si rularea serverului Flask.")
else:
    selected_article_header = st.selectbox(
        "Selectati Articolul:",
        options=all_article_headers_with_blank,
        index=0,
        key="sb_article_header_clean"
    )

    paragraph_options_for_selectbox = [""]

    if selected_article_header:
        @st.cache_data
        def get_paragraphs_for_article_dropdown_cached(_article_header_for_cache):
            encoded_article_header = requests.utils.quote(_article_header_for_cache)
            data = make_api_request(f"articles/{encoded_article_header}/paragraphs")
            return [""] + (data.get("paragraphs", []) if data and isinstance(data.get("paragraphs"), list) else [])

        paragraph_options_for_selectbox = get_paragraphs_for_article_dropdown_cached(selected_article_header)

    paragraph_selectbox_key = f"sb_paragraph_id_clean_{selected_article_header if selected_article_header else 'none'}"

    selected_paragraph_id = st.selectbox(
        "Selectati Paragraful (optional):",
        options=paragraph_options_for_selectbox,
        key=paragraph_selectbox_key,
        disabled=not selected_article_header
    )

    if st.button("Cauta Articol/Paragraf", key="btn_search_article_content_clean"):
        if selected_article_header:
            params = {"article_header": selected_article_header}
            if selected_paragraph_id:
                params["paragraph_identifier"] = selected_paragraph_id

            with st.spinner("Se cauta..."):
                response_data = make_api_request("search/article-content", params=params)

            if response_data and "results" in response_data:
                results = response_data["results"]
                if results:
                    st.subheader(
                        f"Rezultate pentru {selected_article_header} {selected_paragraph_id if selected_paragraph_id else '(toate paragrafele)'}")
                    for res in results:
                        st.markdown(
                            f"**Articol {res.get('article', 'N/A')} Paragraf {res.get('paragraph', 'N/A')} (ID: {res.get('id', 'N/A')})**")
                        st.markdown(res.get("text", "Text indisponibil."))
                        st.markdown("---")
                else:
                    st.info("Niciun rezultat gasit pentru selectia curenta.")
            elif response_data is None and selected_article_header: # Check if make_api_request failed
                pass # Error is already displayed by make_api_request
        else:
            st.warning("Va rugam selectati un articol.")

st.markdown("---")

# --- Section 2: Semantic Search (Based on Embeddings) ---
st.header("Cautare Semantica (Bazata pe Embeddings)")
semantic_query_input = st.text_input("Introduceti intrebarea dvs. pentru cautare semantica:",
                                        key="semantic_query_input_clean")

if st.button("Cauta Semantic", key="semantic_search_button_clean"):
    if semantic_query_input:
        with st.spinner("Se analizeaza intrebarea si se cauta (semantic)..."):
            # This alpha value will be explicitly sent to the API
            # Ensure your API endpoint for semantic search can receive and use 'alpha'
            desired_alpha = 0.3
            current_timestamp = int(time.time()) # Cache-busting
            params = {
                "q": semantic_query_input,
                "k": 5,
                "alpha": desired_alpha,
                "_cache_bust": current_timestamp
            }
            response_data = make_api_request("search/semantic", params=params)

        if response_data and "results" in response_data:
            results = response_data["results"]
            alpha_used = response_data.get('alpha_used', 'N/A') # Get alpha reported by API
            st.subheader(f"Rezultate relevante (Cautare Semantica - Alpha Utilizat de API: {alpha_used})")
            if results:
                for i, res in enumerate(results):
                    score = res.get('final_score')
                    sem_score = res.get('semantic_score')
                    overlap_score = res.get('overlap_score')
                    raw_dist = res.get('raw_distance')

                    score_text = f"{score:.4f}" if isinstance(score, float) else str(score) if score is not None else "N/A"
                    sem_score_text = f"{sem_score:.4f}" if isinstance(sem_score, float) else str(sem_score) if sem_score is not None else "N/A"
                    overlap_score_text = f"{overlap_score:.4f}" if isinstance(overlap_score, float) else str(overlap_score) if overlap_score is not None else "N/A"
                    raw_dist_text = f"{raw_dist:.4f}" if isinstance(raw_dist, float) else str(raw_dist) if raw_dist is not None else "N/A"

                    st.markdown(f"**{i + 1}. Articol {res.get('article', 'N/A')} {res.get('paragraph', 'N/A')}**")
                    st.markdown(
                        f"> Scor Final: **{score_text}** (Semantic: {sem_score_text}, Dist: {raw_dist_text}, Overlap: {overlap_score_text})")
                    st.markdown(res.get("text", "Text indisponibil."))
                    if res.get("matched_concepts") and isinstance(res.get("matched_concepts"), list):
                        st.caption(f"Concepte detectate: `{', '.join(res['matched_concepts'])}`")
                    st.markdown("---")
            else:
                st.info("Niciun rezultat semantic relevant gasit.")
        # Error display implicitly handled by make_api_request if response_data is None
    else:
        st.warning("Va rugam introduceti o intrebare pentru cautarea semantica.")

st.markdown("---")

# --- Section 3: Graph-Based Semantic Search ---
st.header("Cautare Bazata pe Graf (Concepte)")
graph_query_input = st.text_input("Introduceti intrebarea dvs. pentru cautare in graf:",
                                     key="graph_query_input_clean")

if st.button("Cauta in Graf", key="graph_search_button_clean"):
    if graph_query_input:
        with st.spinner("Se analizeaza intrebarea si se cauta in graf..."):
            current_timestamp = int(time.time())
            params = {
                "q": graph_query_input,
                "k": 5,
                "_cache_bust": current_timestamp
            }
            response_data = make_api_request("search/graph", params=params)

        if response_data and "results" in response_data:
            results = response_data["results"]
            st.subheader("Rezultate relevante (Cautare Graf):")
            if results:
                for i, res in enumerate(results):
                    score = res.get('graph_score')
                    score_text = f"{score:.4f}" if isinstance(score, float) else str(score) if score is not None else "N/A"
                    st.markdown(
                        f"**{i + 1}. Articol {res.get('article', 'N/A')} {res.get('paragraph', 'N/A')}** (Scor Graf: {score_text})")
                    st.markdown(res.get("text", "Text indisponibil."))
                    st.markdown("---")
            else:
                st.info("Niciun rezultat relevant gasit in graf.")
    else:
        st.warning("Va rugam introduceti o intrebare pentru cautarea in graf.")

st.markdown("---")

# --- NEW Section 4: Hybrid (Combined) Search ---
st.header("Cautare Hibrida (Combinata Semantic + Graf)")

combined_query_input = st.text_input("Introduceti intrebarea dvs. pentru cautare hibrida:", key="combined_query_input")

if st.button("Cauta Hibrid", key="combined_search_button"):
    if combined_query_input:
        with st.spinner("Se ruleaza ambele cautari si se combina rezultatele..."):
            params = {
                "q": combined_query_input,
                "k": 5,
                "k_candidates": 20
            }
            response_data = make_api_request("search/combined", params=params)

        if response_data and "results" in response_data:
            results = response_data["results"]
            st.subheader("Rezultate relevante (Cautare Hibrida):")
            if results:
                for i, res in enumerate(results):
                    rrf_score = res.get('rrf_score')
                    score_text = f"{rrf_score:.4f}" if isinstance(rrf_score, float) else str(
                        rrf_score) if rrf_score is not None else "N/A"

                    found_by_text = ", ".join(res.get('found_by', []))

                    st.markdown(f"**{i + 1}. Articol {res.get('article', 'N/A')} {res.get('paragraph', 'N/A')}**")
                    st.markdown(f"> Scor Hibrid (RRF): **{score_text}** | Gasit de: *{found_by_text}*")
                    st.markdown(res.get("text", "Text indisponibil."))
                    st.markdown("---")
            else:
                st.info("Niciun rezultat relevant gasit prin cautare hibrida.")
    else:
        st.warning("Va rugam introduceti o intrebare pentru cautarea hibrida.")