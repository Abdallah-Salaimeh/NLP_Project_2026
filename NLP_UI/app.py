import streamlit as st
from transformers import AutoTokenizer
import re
import torch
import torch.nn as nn

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Arabic Text Summarization",
    page_icon="🧠",
    layout="centered"
)

# ── Seed ─────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH    = "seq2seq_model.pth"
EMBEDDING_DIM = 128
HIDDEN_SIZE   = 256
MAX_LEN_SRC   = 256
MAX_SUMMARY   = 63

BOS_ID = 0
EOS_ID = 2
PAD_ID = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Load model — raw layers, no wrapper class, identical to the notebook ─────
@st.cache_resource
def load_model():
    tokenizer  = AutoTokenizer.from_pretrained("moussaKam/AraBART")
    VOCAB_SIZE = tokenizer.vocab_size

    # Build layers exactly as training did — plain nn.Embedding, no padding_idx
    encoder_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM).to(DEVICE)
    encoder_lstm      = nn.LSTM(EMBEDDING_DIM, HIDDEN_SIZE,
                                batch_first=True, bidirectional=True).to(DEVICE)

    decoder_embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM).to(DEVICE)
    decoder_lstm      = nn.LSTM(EMBEDDING_DIM + HIDDEN_SIZE * 2,
                                HIDDEN_SIZE, batch_first=True).to(DEVICE)

    W1 = nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE).to(DEVICE)
    W2 = nn.Linear(HIDDEN_SIZE,     HIDDEN_SIZE).to(DEVICE)
    V  = nn.Linear(HIDDEN_SIZE,     1).to(DEVICE)
    fc = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    encoder_embedding.load_state_dict(checkpoint["encoder_embedding"])
    encoder_lstm.load_state_dict(checkpoint["encoder_lstm"])
    decoder_embedding.load_state_dict(checkpoint["decoder_embedding"])
    decoder_lstm.load_state_dict(checkpoint["decoder_lstm"])
    W1.load_state_dict(checkpoint["W1"])
    W2.load_state_dict(checkpoint["W2"])
    V.load_state_dict(checkpoint["V"])
    fc.load_state_dict(checkpoint["fc"])

    for m in [encoder_embedding, encoder_lstm,
              decoder_embedding, decoder_lstm,
              W1, W2, V, fc]:
        m.eval()

    return tokenizer, encoder_embedding, encoder_lstm, \
           decoder_embedding, decoder_lstm, W1, W2, V, fc


# ── Cleaning ──────────────────────────────────────────────────────────────────
def cleaning_words(x):
    x = re.sub(r' +', ' ', x)
    x = re.sub(r'"([^"]+)"', r'\1', x, flags=re.UNICODE)
    x = re.sub(r'[»«]', '', x)
    x = re.sub(r'[a-zA-Z]', '', x)
    x = re.sub(r'\(\d+\s*/\s*\w+\)', '', x)
    x = re.sub(r'[()/]', '', x)
    x = re.sub(r'[ء-ي]\.[ء-ي]', '', x)
    x = re.sub(r'\s[ء-ي]\s', ' ', x)
    x = re.sub(r'\n', '', x)
    for ch, rep in [('\xa0',' '),('\u200f',''),('\u200e',''),('\ufeff',''),
                    ('\u200c',''),('\u200b',''),('\u202c',''),('\u202a',''),
                    ('\u202b',''),('…','.'),('٪','%'),('٬',''),('―','-')]:
        x = x.replace(ch, rep)
    return x


# ── Inference — copy-paste of notebook summarize() ───────────────────────────
def generate_summary(text, tokenizer,
                     encoder_embedding, encoder_lstm,
                     decoder_embedding, decoder_lstm,
                     W1, W2, V, fc):

    text = cleaning_words(text)

    enc_tokens  = tokenizer(text, max_length=MAX_LEN_SRC, truncation=True,
                            padding="max_length", return_tensors="pt")
    encoder_ids = enc_tokens["input_ids"].to(DEVICE)
    src_mask    = enc_tokens["attention_mask"].to(DEVICE)

    with torch.no_grad():
        enc_emb = encoder_embedding(encoder_ids)
        encoder_outputs, (h, c) = encoder_lstm(enc_emb)

        h = torch.cat((h[0], h[1]), dim=1).unsqueeze(0).contiguous()
        c = torch.cat((c[0], c[1]), dim=1).unsqueeze(0).contiguous()
        decoder_hidden = (h[:, :, :HIDDEN_SIZE].contiguous(),
                          c[:, :, :HIDDEN_SIZE].contiguous())

        W1_out    = W1(encoder_outputs)
        dec_token = torch.tensor([[BOS_ID]], device=DEVICE)

        generated_ids = []

        for _ in range(MAX_SUMMARY):
            dec_emb      = decoder_embedding(dec_token)
            hidden_state = decoder_hidden[0].squeeze(0)

            score = V(torch.tanh(W1_out + W2(hidden_state).unsqueeze(1)))
            score = score.masked_fill(src_mask.unsqueeze(2) == 0, -1e9)
            attn    = torch.softmax(score, dim=1)
            context = torch.sum(attn * encoder_outputs, dim=1).unsqueeze(1)

            lstm_input = torch.cat((dec_emb, context), dim=2)
            h_s, c_s   = decoder_hidden
            decoder_hidden = (h_s.contiguous(), c_s.contiguous())
            dec_output, decoder_hidden = decoder_lstm(lstm_input, decoder_hidden)

            next_token = fc(dec_output.squeeze(1)).argmax(dim=-1).item()

            if next_token == EOS_ID: break
            if next_token == PAD_ID: break

            # Only stop if the last 4 tokens are all the same (true hard loop)
            if len(generated_ids) >= 4 and all(t == next_token for t in generated_ids[-4:]):
                break

            generated_ids.append(next_token)
            dec_token = torch.tensor([[next_token]], device=DEVICE)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧠 Arabic Text Summarization")
st.markdown("Paste an Arabic news article below and click **Generate Summary**.")

try:
    (tokenizer, encoder_embedding, encoder_lstm,
     decoder_embedding, decoder_lstm, W1, W2, V, fc) = load_model()
    st.success("Model loaded!", icon="✅")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

article = st.text_area(
    "Enter Arabic Article:",
    height=300,
    placeholder="الصق المقال العربي هنا..."
)

if st.button("Generate Summary", type="primary"):
    if article.strip() == "":
        st.warning("Please enter some Arabic text first.")
    else:
        with st.spinner("Generating summary..."):
            summary = generate_summary(
                article, tokenizer,
                encoder_embedding, encoder_lstm,
                decoder_embedding, decoder_lstm,
                W1, W2, V, fc
            )
        st.subheader("📝 Generated Summary")
        st.success(summary)
