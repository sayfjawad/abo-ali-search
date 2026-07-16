# Abo Ali Smart Search

[English](README.md) | **Nederlands** | [العربية](README.ar.md)

**AI-gedreven semantisch zoeken en RAG-vraagbeantwoording over een groot
archief van getranscribeerde Arabische video's — volledig lokaal,
GPU-versneld, met tijd-geïndexeerde, afspeelbare resultaten.**

Een factcheck-project: elke publieke uitspraak van een zelfverklaarde
"genezer" doorzoekbaar maken, zodat zijn eigen woorden getoetst kunnen worden
aan elkaar en aan de werkelijkheid. Zie
[Waarom dit project bestaat](#1-waarom-dit-project-bestaat).

Live: <https://aboali.scrib-r.com>

---

## 1. Waarom dit project bestaat

Dit is een factcheck-project. Het onderwerp, **Abu Ali Al-Shaibani**, is een
zelfverklaarde helderziende die beweert berichten en instructies te ontvangen
van zijn "leraar", en die claimt mensen te kunnen genezen van zo ongeveer
alles — met speciale medicijnen die hij voor duizenden euro's verkoopt aan
nietsvermoedende kijkers van zijn tv-show, zijn website en zijn
YouTube-kanaal. Ook mijn dierbaren zijn slachtoffer geworden van zijn
praktijken.

Discussiëren met hem of zijn volgers heeft geen zin; dat is inmiddels wel
gebleken. Dus kiest dit project een andere aanpak: **laat de man zichzelf
tegenspreken**. Verzamel alles wat hij ooit heeft gezegd en beweerd, in alle
media-formaten, en maak dat voor iedereen doorzoekbaar. Als elke voorspelling,
claim en tegenstrijdigheid binnen seconden te vinden is — met een
afspeelbare videoclip met tijdstempel als bewijs — is discussie niet meer
nodig.

Het plan, in zes stappen:

1. Alles vinden wat hij ooit heeft gezegd, in alle media-formaten en bronnen.
2. Al zijn uitspraken omzetten naar digitale tekst.
3. Een data-library bouwen met zijn video's, stem en transcripties.
4. De library doorzoekbaar maken met AI-technieken (tokenization,
   vectorization, RAG).
5. Een online applicatie bouwen zodat *iedereen* zijn uitspraken kan
   doorzoeken.
6. Alles vrijgeven, zodat de feiten voor zich spreken.

Deze repository is stap 4–6: het zoekportaal over het getranscribeerde
archief. De transcriptie-pipeline die het archief voedt (stap 2–3) is
uitgegroeid tot een eigen product: [scrib-r](https://www.scrib-r.com), een
transcriptieportaal dat video en audio omzet naar tekst (gratis zolang het in
alfa is).

Dezelfde opzet werkt om *elke* publieke figuur aan zijn eigen woorden te
houden. Ken je andere charlatans die misbruik maken van mensen? Dan help ik
graag om ook die aan te pakken.

---

## 2. Wat de applicatie is

Deze applicatie maakt een video-archief van ~2.500 afleveringen (2015–2026,
≈500 GB aan media plus transcripties met sprekerherkenning) *doorzoekbaar op
betekenis*. In plaats van te matchen op trefwoorden wordt elk
transcriptsegment ingebed in een meertalige vectorruimte, zodat een vraag in
je eigen woorden de momenten vindt waarop de spreker dat onderwerp
daadwerkelijk besprak — ook als de bewoording verschilt.

Eén Arabische (RTL) webpagina biedt twee interactiemodi:

| Modus | Wat er gebeurt |
|---|---|
| **🔍 بحث فقط** (semantisch zoeken) | De zoekvraag wordt op de GPU ingebed en gematcht tegen ~160.000 transcript-chunks. Resultaten verschijnen in een tabel: **datum** van de aflevering, **tijdstip in de video**, de gevonden **gemarkeerde tekst** en de **titel** van de aflevering — gesorteerd op relevantie en filterbaar op datumbereik. |
| **🤖 اسأل الذكاء الاصطناعي** (Vraag de AI / RAG) | De best passende fragmenten gaan naar een **lokale LLM**, die een Arabisch antwoord met bronvermelding schrijft over *wat er is gezegd en wanneer*. Citaties zoals `[3]` in het antwoord linken door naar de bijbehorende cliprij. |

Elke resultaatrij biedt twee manieren om *het exacte moment te bekijken*:

- **▶ تشغيل** — speelt het lokale mediabestand af in een speler op de pagina,
  automatisch gestart op ~2 seconden vóór de geciteerde passage (geserveerd
  met HTTP Range-ondersteuning, dus spoelen in bestanden van meerdere GB's is
  direct).
- **YouTube ↗** — een deeplink van de vorm
  `https://www.youtube.com/watch?v=<id>&t=<seconds>s` die hetzelfde moment
  online opent; geschikt om te delen met eindgebruikers.

Alles — embedding, retrieval, antwoordgeneratie, mediastreaming — draait op
de lokale machine. Er zijn geen cloud-API's bij betrokken en er verlaat geen
data de host.

---

## 3. Architectuur

```
                       ┌──────────────────────────────────────────────┐
                       │  FastAPI-app (uvicorn, poort 8901)           │
 Browser (RTL SPA) ───▶│                                              │
   static/index.html   │  /api/search ──▶ BGE-M3 query-embedding      │
                       │                  cosine top-k op GPU          │
                       │                  (~160k × 1024 fp16-matrix)   │
                       │                                              │
                       │  /api/ask ─────▶ retrieval (als hierboven)   │
                       │                  → OpenAI-compatibele lokale │
                       │                    LLM (llama.cpp / Qwen3)   │
                       │                  → Arabisch antwoord met     │
                       │                    citaties                  │
                       │                                              │
                       │  /media/{file} ─▶ Range-enabled streaming    │
                       │                   van lokale mp4/m4a         │
                       └──────────────────────────────────────────────┘
                                        ▲
                     build_index.py (offline, eenmalig / bij nieuwe video's)
                     parseert transcripties → chunks → GPU-embeddings
                     → index/index.sqlite + index/embeddings.npy
```

### Retrieval-ontwerp

- **Chunking** — opeenvolgende segmenten van dezelfde spreker worden
  samengevoegd tot ~700 tekens (`MERGE_TARGET_CHARS` in `build_index.py`),
  met behoud van de oorspronkelijke start-/eindtijden. Zo krijgt elke vector
  genoeg context terwijl de tijdstempels precies blijven. ~2.500 video's →
  ~160.000 chunks.
- **Embeddings** — [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)
  (dense modus, 1024-d, fp16), een van de sterkste open meertalige
  retrieval-modellen voor Arabisch. Direct geladen via `transformers`
  (CLS-pooling + L2-normalisatie) — geen zwaar framework nodig.
- **Vectorzoeken** — bewust *geen* vectordatabase. De hele matrix
  (160k × 1024 fp16 ≈ 320 MB) staat op de GPU en een query is één matmul:
  enkele milliseconden, exact (geen ANN-benadering), nul infrastructuur. Dit
  schaalt comfortabel tot miljoenen chunks voordat FAISS/qdrant zich
  terugbetaalt.
- **Datumfiltering** — elke chunk draagt de uploaddatum van zijn aflevering;
  filters worden toegepast door scores te maskeren vóór de top-k.

### RAG-antwoordgeneratie

`/api/ask` bouwt een prompt van genummerde fragmenten (titel, datum,
tijdstip, tekst) plus de vraag van de gebruiker, en roept **elk
OpenAI-compatibel chat-endpoint** aan. Standaard wordt een
llama.cpp-container met de naam `scrib-r-backend-llama-1` (Qwen3-8B)
automatisch gevonden via `docker inspect`; te overschrijven met
omgevingsvariabelen (zie Configuratie). `<think>…</think>`-blokken van
redeneermodellen worden verwijderd; citatiemarkeringen `[n]` in het antwoord
worden gekoppeld aan de brontabel.

---

## 4. Technologieën

| Laag | Technologie | Waarom |
|---|---|---|
| Embeddings | `BAAI/bge-m3` via 🤗 `transformers`, PyTorch, CUDA fp16 | Top open meertalige (incl. Arabisch) dense retrieval; draait op één V100 |
| Vectorzoeken | PyTorch-matmul op GPU | Exact, ~ms latency op deze schaal, geen extra service |
| Metadata-opslag | SQLite | Single-file, zero-ops opslag voor video- en chunk-metadata |
| Backend | FastAPI + uvicorn | Async Python-API, statische bestanden, Range-capabele `FileResponse` |
| Antwoord-LLM | Elke OpenAI-compatibele server (llama.cpp, LM Studio, Ollama, vLLM) | Provider-onafhankelijk; standaard een llama.cpp-container met Qwen3-8B |
| Frontend | Vanilla HTML/CSS/JS, RTL, dark-mode-aware, responsive (mobiele kaartweergave) | Geen build-stap, geen framework — één bestand |
| Deployment | systemd-service + nginx reverse proxy + Let's Encrypt (certbot) | Start mee bij boot, TLS, publieke hostnaam |

---

## 5. Verwachte data-indeling

De indexer leest een platte map (standaard `/data/ABO_ALI`) met, per video:

```
<YYYYMMDD>_<youtube-id>.metadata.json   # titel, url, upload_date, duur
<YYYYMMDD>_<youtube-id>.json            # transcript: segments[] met
                                        #   speaker_id, speaker, start, end, text
<YYYYMMDD>_<youtube-id>.mp4             # (of .m4a) lokale media, optioneel
```

Voorbeeld van `metadata.json`:

```json
{
  "id": "MwBBTTMiS5s",
  "title": "…",
  "url": "https://www.youtube.com/watch?v=MwBBTTMiS5s",
  "upload_date": "20150110",
  "duration_seconds": 4131.0
}
```

Transcripties in dit formaat komen uit een WhisperX + pyannote-pipeline
(bijv. [scrib-r](https://github.com/sayfjawad/scrib-r)); elke bron werkt
zolang de JSON-vorm overeenkomt.

---

## 6. Installatie

### Vereisten

- Linux, Python 3.10+
- NVIDIA GPU met ≥ 8 GB VRAM (getest: Tesla V100 32 GB) + CUDA-enabled PyTorch
- ~500 MB schijfruimte voor de index (bij 160k chunks)
- Optioneel: een draaiende OpenAI-compatibele LLM-server voor de Vraag-AI-modus
- Optioneel: Docker (alleen voor auto-discovery van de standaard llama.cpp-container)

### Hardware-richtlijnen voor de volledige pipeline

Wil je zelf zo'n archief bouwen, van begin tot eind? Grofweg, op basis van
dit project:

**Doorzoekbare library + lokale LLM (deze repo):**

- GPU met minimaal 16 GB VRAM — 24 GB kan al goed, 32 GB (bijv. een RTX 5090)
  is ideaal om zowel het embedding-model als een antwoord-LLM comfortabel te
  draaien
- Een hedendaagse CPU (i5 / Ryzen 5 of nieuwer)
- Een internetverbinding om de site te hosten

**Transcriberen (upstream, bijv. [scrib-r](https://github.com/sayfjawad/scrib-r)):**

- GPU met 8 GB VRAM en 16 GB RAM
- Een moderne CPU (i5 / Ryzen 5 of hoger aanbevolen)
- Voldoende SSD-ruimte voor de media die je verzamelt — dit archief gebruikt
  ~500 GB

Met deze setup — en desnoods een online AI ernaast — kun je zelf een shady
publiek figuur onder de microscoop leggen.

### Installeren

```bash
git clone git@github.com:sayfjawad/abo-ali-search.git
cd abo-ali-search
python3 -m venv --system-site-packages .venv   # hergebruik een systeem-CUDA-torch indien aanwezig
.venv/bin/pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" transformers torch numpy httpx
```

> Als je systeem-Python al een CUDA-build van `torch` heeft (zoals op
> ML-hosts), voorkomt `--system-site-packages` een tweede multi-GB-installatie.
> Op een schone machine: laat de vlag weg en laat pip torch installeren.

### De index bouwen

```bash
# pas DATA_DIR in build_index.py aan als je archief ergens anders staat
CUDA_VISIBLE_DEVICES=0 .venv/bin/python build_index.py
```

De eerste run downloadt het BGE-M3-model (~2,3 GB) naar `HF_HOME`. Het
indexeren van het volledige corpus van 2.500 video's duurt ~8 minuten op een
V100. Draai opnieuw wanneer er nieuwe video's bijkomen (de index wordt
volledig opnieuw opgebouwd; dat is goedkoop).

### Draaien

```bash
./run.sh                # http://localhost:8901
```

### Als service draaien (systemd)

```bash
sudo cp deploy/aboali-search.service /etc/systemd/system/
# pas User/WorkingDirectory/paden aan als die afwijken
sudo systemctl daemon-reload
sudo systemctl enable --now aboali-search
```

### Publieke HTTPS via nginx + certbot (optioneel)

Op de internet-facing host (mag een andere machine zijn die de app bereikt
over een VPN/overlay-netwerk):

```bash
sudo cp deploy/nginx-aboali.conf /etc/nginx/sites-available/aboali.example.com
# pas server_name en het proxy_pass-doel aan
sudo ln -s /etc/nginx/sites-available/aboali.example.com /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d aboali.example.com --redirect
```

`proxy_buffering off` en lange timeouts in de meegeleverde config zijn
belangrijk voor soepele videostreaming door de proxy.

> ⚠️ De app zelf heeft **geen authenticatie**. Als je hem publiek maakt, kan
> iedereen zoeken, de mediabestanden streamen en het (GPU-verbruikende)
> Vraag-AI-endpoint gebruiken. Scherm af met nginx basic-auth, een allowlist
> of een auth-proxy als dat nodig is.

---

## 7. Configuratie

Alles optioneel, via omgevingsvariabelen (zie `run.sh` / de systemd-unit):

| Variabele | Standaard | Doel |
|---|---|---|
| `HF_HOME` | `/data/huggingface` | Locatie van de Hugging Face-modelcache |
| `CUDA_VISIBLE_DEVICES` | `0` | Welke GPU queries embedt / de matrix vasthoudt |
| `LLM_BASE_URL` | auto-discovery van container `scrib-r-backend-llama-1` | OpenAI-compatibel endpoint voor Vraag-AI, bijv. `http://localhost:1234/v1` |
| `LLM_MODEL_ID` | `qwen3-8b` | Modelnaam die aan het endpoint wordt doorgegeven |
| `LLM_API_KEY` | `none` | Bearer-token als het endpoint er een vereist |

Paden die in code staan: archiefmap (`DATA_DIR` in `build_index.py`,
`MEDIA_DIR` in `app.py`) en chunk-samenvoeggrootte (`MERGE_TARGET_CHARS`).

---

## 8. HTTP-API

| Endpoint | Methode | Body / parameters | Retourneert |
|---|---|---|---|
| `/api/search` | POST | `{query, top_k?, date_from?, date_to?}` (datums `YYYYMMDD`) | `{results: [{score, video, title, date, start, end, start_fmt, text, youtube_url, media_url, …}]}` |
| `/api/ask` | POST | `{question, top_k?, date_from?, date_to?}` | `{answer, error, sources: [...]}` — sources dragen `cited: true` waar geciteerd |
| `/api/stats` | GET | — | `{videos, chunks}` |
| `/media/{file}` | GET | Range ondersteund | mp4/m4a-stream |
| `/` | GET | — | de single-page UI |

Voorbeeld:

```bash
curl -s localhost:8901/api/search -H 'Content-Type: application/json' \
  -d '{"query":"ماذا قال عن الزلازل في تركيا","top_k":5,"date_from":"20160101"}'
```

---

## 9. Prestaties (referentiehardware: Tesla V100 32 GB)

| Bewerking | Gemeten |
|---|---|
| Volledige index-build (2.523 video's → 159.532 chunks) | ~8 min |
| Semantische zoekopdracht (embed + exacte top-k) | < 100 ms |
| RAG-antwoord (Qwen3-8B, 12 fragmenten) | ~4–5 s |
| GPU-geheugen (serving) | ~1,8 GB |

---

## 10. Repository-indeling

```
app.py                  FastAPI-backend (search, ask, media, stats)
build_index.py          offline indexer (parse → chunk → embed → store)
embedder.py             BGE-M3-embedding-wrapper (transformers, fp16)
static/index.html       de volledige frontend (RTL, responsive, dark-mode)
run.sh                  dev/foreground-launcher
deploy/
  aboali-search.service systemd-unit
  nginx-aboali.conf     reverse-proxy-template (TLS via certbot)
index/                  gegenereerd — niet gecommit (sqlite + npy)
```

---

## 11. Steun dit project

De website, de informatie en de hardware kosten tijd en geld. Elke vorm van
hulp is welkom:

**[Doneer via GoFundMe — help dit archief in de lucht te houden](https://www.gofundme.com/f/factchecken-met-ai-help-dit-archief-in-de-lucht-te-houden)**

Andere manieren om te helpen: meld nieuwe video's of bronnen die in het
archief ontbreken, verbeter de code via issues en pull requests, of gebruik
dit project als blauwdruk om een andere charlatan te factchecken — ik help
je graag op weg.
