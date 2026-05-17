DATASET_NAME = "KisanVaani/agriculture-qa-english-only"

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
MAX_SENTENCES = 4
OVERLAP_SENTENCES = 1

TOP_K_SEMANTIC = 20
TOP_K_BM25 = 20
RRF_K = 60
FINAL_TOP_K = 5

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR_MODEL_NAME = "google/flan-t5-base"

MAX_CONTEXT_WORDS = 450
MAX_NEW_TOKENS = 220

EVAL_SIZE = 50
K_VALUES = [1, 3, 5]

EMBEDDING_MODELS = {
    "BGE-Small": EMBEDDING_MODEL_NAME,
    "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
}

CROSS_ENCODER_BASE = CROSS_ENCODER_MODEL_NAME
CROSS_ENCODER_SAVE_DIR = "models/agri_cross_encoder"

EMBEDDINGS_SAVE_PATH = "models/agri_embeddings.npy"

GENERATOR_MODEL = GENERATOR_MODEL_NAME
MAX_CONTEXT_TOKENS = MAX_CONTEXT_WORDS

EMBEDDING_BATCH_SIZE = 32
EMBEDDING_SAMPLE_SIZE = 1000

FINETUNE_BATCH_SIZE = 8
FINETUNE_EPOCHS = 1

EVAL_TEST_SIZE = EVAL_SIZE
GEN_EVAL_SIZE = 20

TEST_QUERIES = [
    "How to control pests in cotton?",
    "Which fertilizer is good for wheat?",
    "Why are tomato leaves turning yellow?",
    "How to improve soil fertility?",
    "How to control fungal disease in rice?",
]

DOMAIN_STOPWORDS = {
    "please", "tell", "give", "use", "used", "using",
    "crop", "plant", "plants", "field", "farmer",
    "agriculture", "agricultural"
}

TOPIC_KEYWORDS = {
    "fertilizer": ["fertilizer", "nutrient", "npk", "urea", "dap", "manure"],
    "pest": ["pest", "insect", "aphid", "whitefly", "borer", "mites"],
    "disease": ["disease", "fungal", "bacterial", "virus", "blight", "rot"],
    "soil": ["soil", "ph", "organic", "compost", "drainage"],
    "irrigation": ["water", "irrigation", "drip", "moisture", "drought"],
}

TOPIC_COLORS = {
    "fertilizer": "#2E86AB",
    "pest": "#F18F01",
    "disease": "#C73E1D",
    "soil": "#6A994E",
    "irrigation": "#3A86FF",
    "other": "#777777",
}

TOPIC_QUERIES = {
    "fertilizer": "Which fertilizer should be used for better crop growth?",
    "pest": "How can I control insect pests on my crop?",
    "disease": "How to treat fungal disease in plants?",
    "soil": "How to improve soil fertility?",
    "irrigation": "How often should I irrigate my crop?",
}



