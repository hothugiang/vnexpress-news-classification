
Dưới đây là phân tích chi tiết UH-CSG-CRS và so sánh với các phương pháp SOTA về CRS dựa trên knowledge.

## 1. Bản chất kiến trúc của UH-CSG-CRS

UH-CSG-CRS không phải một framework CRS hoàn toàn mới — nó là một **module-level intervention** nhắm chính xác vào phần yếu nhất của pipeline MSCRS/UniCRS: cách graph encode collaborative signal cho items. Cụ thể, nó thay thế `co_occurrence graph` + 3 GCNConv song song bằng một unified heterogeneous graph duy nhất, sau đó dùng LightGCN propagation thay vì GCNConv có weight matrix.

Điểm thiết kế đáng chú ý nhất là mô hình coi mỗi **dialogue session** như một "pseudo-user node" trong bipartite graph. Đây là ý tưởng mượn từ collaborative filtering truyền thống (NGCF, LightGCN) nhưng áp dụng vào bối cảnh CRS — nơi không có explicit user ID xuyên suốt các sessions. Dialogue session $d_j$ kết nối với tất cả items được recommend trong session đó qua edges $\mathcal{E}_{DI}$, tạo thành một implicit CF signal mà không cần user identity.

## 2. Vấn đề gốc rễ mà UH-CSG nhắm vào

UH-CSG phát hiện một vấn đề mà hầu hết papers CRS trước đó không articulate rõ: **test-only item cold-start**. Trong datasets CRS (ReDial, INSPIRED), có overlap giữa item spaces của train/val/test splits. Items chỉ xuất hiện ở test set không có edges nào trong co-occurrence graph (vì co-occurrence chỉ được build từ train), nên chúng hoàn toàn "mù" về collaborative signal. Tất cả các methods trước — từ KBRD, KGSF, C²-CRS, cho đến UniCRS và MSCRS — đều mắc phải vấn đề này mà không nhận ra.

UH-CSG giải quyết bằng **semantic-bridge paths**: item test-only vẫn có text/image similarity edges ($\mathcal{E}_{TS}$, $\mathcal{E}_{IS}$) nối tới items trong train set, và từ đó "nhảy" qua dialogue nodes để nhận collaborative signal gián tiếp. Path 2-hop điển hình:

$$i_{\text{test}} \xrightarrow{\mathcal{E}_{TS}} i_{\text{train}} \xrightarrow{\mathcal{E}_{DI}} d_j$$

Đây là information path mà 3 graphs độc lập của MSCRS không thể tạo ra.

## 3. So sánh với các SOTA knowledge-based CRS

### 3.1 vs. KBRD (Chen et al., 2019)

KBRD dùng DBpedia KG + GCN để encode entity representations, rồi dùng chúng bias recommender. KBRD chỉ dùng **structural relations** từ KG — không có semantic similarity, không có collaborative signal ngoài conversation history. Mỗi entity representation hoàn toàn phụ thuộc vào topology của DBpedia. UH-CSG bổ sung 4 loại edge (DI, UI, TS, IS) nên entity representations giàu hơn đáng kể. Tuy nhiên, KBRD có lợi thế là đơn giản và không cần external data (MovieLens).

### 3.2 vs. KGSF (Zhou et al., 2020)

KGSF kết hợp **hai KGs** — DBpedia (factual) và ConceptNet (commonsense) — thông qua mutual information maximization để align hai semantic spaces. So với UH-CSG, KGSF thiếu collaborative signal hoàn toàn; nó dựa trên semantic reasoning qua KG paths. KGSF xử lý entity linking tốt hơn nhờ ConceptNet nhưng không có cơ chế nào cho test-only cold-start. Điểm khác biệt lớn: KGSF dùng word-level KG (ConceptNet) để bridge lexical gap, trong khi UH-CSG dùng content similarity + CF edges để bridge item-level gap. Hai hướng bổ trợ chứ không mâu thuẫn.

### 3.3 vs. C²-CRS (Zhou et al., 2022)

C²-CRS dùng contrastive learning để align representations giữa conversation context và KG. Nó tạo coarse-grained và fine-grained views rồi dùng InfoNCE loss để kéo lại gần nhau. C²-CRS giải quyết vấn đề **representation gap** giữa dialogue encoder và KG encoder — một vấn đề khác với cold-start item problem mà UH-CSG nhắm vào. C²-CRS vẫn dùng DBpedia graph tĩnh, nên items test-only vẫn thiếu collaborative signal.

### 3.4 vs. UniCRS (Wang et al., 2022)

UniCRS thống nhất recommendation và conversation generation vào một prompt learning framework duy nhất với PLM backbone. UniCRS dùng R-GCN trên DBpedia để encode entities, rồi inject vào soft prompts qua cross-attention. UH-CSG **xây trên nền UniCRS** — giữ nguyên prompt framework, cross-attention, training pipeline — chỉ thay thế entity embedding module. Nói cách khác, UniCRS là backbone mà UH-CSG extend, không phải competitor trực tiếp. Cải tiến của UH-CSG nằm hoàn toàn ở khâu entity representation quality.

### 3.5 vs. MSCRS (Wei et al., 2025)

Đây là baseline trực tiếp nhất. MSCRS extend UniCRS bằng 3 parallel semantic graphs (text_sim, image_sim, co_occurrence), mỗi graph chạy GCNConv riêng rồi **cộng trực tiếp** outputs. UH-CSG thay thế cả 3 GCNConv bằng 1 LightGCN trên unified graph. Khác biệt cốt lõi:

- MSCRS: signal bị "giam" trong từng graph — text similarity không bao giờ "biết" co-occurrence pattern và ngược lại.
- UH-CSG: cross-edge paths cho phép signal lan qua ranh giới giữa semantic và collaborative. Một item có text similarity cao với item phổ biến sẽ nhận được collaborative boost.
- MSCRS: co_occurrence graph chỉ build từ train → test-only items không có edges.
- UH-CSG: text/image similarity edges cover **tất cả** items → bridge cho test-only items.

### 3.6 vs. VRICR (Zhao et al., 2022)

VRICR dùng variational reasoning trên incomplete KG để xử lý missing relations trong DBpedia. Nó model uncertainty của KG paths bằng variational inference. VRICR giải quyết **KG incompleteness** (missing relations/entities), trong khi UH-CSG giải quyết **collaborative signal absence** cho cold-start items. Hai vấn đề khác chiều — VRICR có thể bổ sung UH-CSG nếu KG có nhiều missing links.

### 3.7 vs. DC-MoME-CRS (trong project knowledge của bạn)

DC-MoME-CRS tập trung vào **dynamic modality fusion** — routing giữa KG/text/visual modalities dựa trên dialogue context tại mỗi turn. Nó giải quyết câu hỏi "modality nào quan trọng nhất cho entity này tại turn này?", trong khi UH-CSG giải quyết "làm sao entity representation có đủ collaborative signal?". Hai contributions orthogonal — có thể kết hợp DC-MoME routing trên entity embeddings đã được UH-CSG enrich.

## 4. Đánh giá điểm mạnh

**Elegance trong thiết kế:** UH-CSG dùng LightGCN (không có learnable weight matrix) thay vì GCNConv, giảm parameters trong khi vẫn tăng expressiveness nhờ graph structure giàu hơn. Gated fusion $\gamma_k$ cho phép model tự học cân bằng giữa content và CF signal cho từng item — items có nhiều collaborative evidence sẽ lean về CF, items cold-start sẽ lean về content.

**Drop-in replacement:** Output `h_entity` giữ nguyên shape và dtype so với MSCRS, nên toàn bộ downstream pipeline (prompt construction, recommendation head, generation) không cần thay đổi. Đây là engineering advantage lớn khi integrate.

**Principled cold-start solution:** Thay vì dùng heuristics (e.g., fallback to content-only cho cold items), UH-CSG cho phép cold items tự nhiên nhận signal qua graph propagation. Không cần special-case handling.

## 5. Đánh giá điểm yếu / rủi ro

**Scalability của adjacency matrix:** Matrix $(N+M+K) \times (N+M+K)$ với $N$ = num_entities (hàng chục nghìn), $M$ = training dialogues (hàng nghìn), $K$ = MovieLens users (có thể hàng chục nghìn) sẽ lớn đáng kể. Dù sparse, LightGCN 4 layers trên graph này tốn computation. Cần benchmark memory/time vs. 3 GCNConv nhỏ riêng lẻ.

**Dependency vào MovieLens:** Edge type $\mathcal{E}_{UI}$ yêu cầu external dataset (MovieLens) có item overlap với DBpedia entities. Điều này giới hạn applicability — nếu domain không phải movies (e.g., books, music), cần tìm external CF source khác. Claim "không thay đổi input format" đúng về inference, nhưng graph construction phụ thuộc data availability.

**Dialogue-as-user assumption:** Model hóa dialogue session $d_j$ như user node giả định mọi items trong cùng session share "user intent" — nhưng trong thực tế, một dialogue có thể chứa items từ nhiều topics khác nhau (user thay đổi preference mid-conversation). Mean pooling cho $\mathbf{h}_{d_j}^{(0)}$ sẽ blur signal nếu session heterogeneous.

**Chưa address generation quality:** UH-CSG chỉ thay đổi entity embeddings — recommendation accuracy có thể tăng nhờ CF signal, nhưng response generation quality phụ thuộc vào soft prompt framework mà UH-CSG không modify. Cần experiment cả BLEU/generation metrics, không chỉ Recall/NDCG.

**Novelty claim N1 cần nuance:** "Dialogue-as-user" trong bipartite CF graph cho CRS là mới, nhưng ý tưởng coi implicit interactions (sessions, clicks) như user proxy đã tồn tại trong session-based recommendation (SR-GNN, FGNN). Novelty nằm ở application context (CRS) chứ không phải ở technique level.

## 6. Tổng kết vị trí trong landscape

UH-CSG-CRS đóng góp một insight quan trọng: **collaborative signal và semantic signal trong CRS nên sống trong cùng một graph** thay vì parallel graphs. Insight này đơn giản nhưng principled, và vấn đề test-only cold-start mà nó identify là genuine gap trong literature. Tuy nhiên, nó là incremental improvement trên MSCRS/UniCRS pipeline chứ không phải paradigm shift — nó không thay đổi cách CRS reason over dialogues, không thay đổi generation mechanism, và yêu cầu external data source (MovieLens) mà không phải domain nào cũng có. Sức mạnh thực sự sẽ phụ thuộc vào empirical results: liệu unified graph có thực sự cải thiện đáng kể Recall@k cho test-only items so với MSCRS baseline hay không.





# UH-CSG Integration Guide
## Chuyển đổi MSCRS → UH-CSG-CRS

---

## 1. Tổng quan các file mới

```
uhcsg_graph_builder.py   # Build unified CSG graph
model_uhcsg.py           # UHCSGPrompt (thay thế MMPrompt)
prepare_movielens.py     # Xử lý MovieLens data thực
train_pre_redial_uhcsg.py # Ví dụ pretrain script
```

## 2. Những gì thay đổi vs. MSCRS

### model_prompt.py → model_uhcsg.py

| Component | MSCRS (MMPrompt) | UH-CSG (UHCSGPrompt) |
|-----------|------------------|----------------------|
| `conv_c1/c2/c3` | 3 CustomGCNConv trên co-occurrence graph | **Xóa** |
| `conv_ts1/ts2/ts3` | 3 CustomGCNConv trên text_sim graph | **Xóa** |
| `conv_is1/is2/is3` | 3 CustomGCNConv trên image_sim graph | **Xóa** |
| `edge_index_c` | Co-occurrence edges | **Xóa** |
| `edge_index_t_s` | Text sim edges (local space) | → Merged vào unified graph |
| `edge_index_i_s` | Image sim edges (local space) | → Merged vào unified graph |
| `sorted_indices`, `idx_to_id_tensor` | Map local↔global | **Xóa** (unified graph dùng global space) |
| **Mới**: `unified_edge_index` | — | Unified CSG edges |
| **Mới**: `lightgcn_conv` | — | 1 CustomGCNConv instance, chạy L lần |
| **Mới**: `fusion_gate` | — | `nn.Linear(2d, 1)` cho gated fusion |
| **Mới**: dialogue/user node init | — | Padded tensors + mask |
| `get_entity_embeds()` | 9 conv layers → index_add → proj | R-GCN → init nodes → LightGCN → gate → proj |
| `forward()` | Interface giữ nguyên | Interface giữ nguyên |
| Output shape | `[N, hidden_size]` | `[N, hidden_size]` (identical) |

### Training scripts: Thay đổi tối thiểu

Chỉ cần thay đổi **3 chỗ** trong mỗi training script:

#### Thay đổi 1: Import
```python
# TRƯỚC (MSCRS)
from model_prompt import MMPrompt

# SAU (UH-CSG)
from model_uhcsg import UHCSGPrompt
from uhcsg_graph_builder import UHCSGGraphBuilder, create_mock_dialogue_items, create_mock_movielens_edges
```

#### Thay đổi 2: Build unified graph (thêm sau data loading)
```python
# === THÊM MỚI: Build UH-CSG graph ===
# Mock data (để test pipeline)
dialogue_items = create_mock_dialogue_items(kg['item_ids'], n_dialogues=500)
ml_edges, n_ml_users = create_mock_movielens_edges(kg['item_ids'], n_users=100)

builder = UHCSGGraphBuilder(
    n_entity=kg['num_entities'],
    edge_index_t_s=text_simi['edge_index_t_s'],
    edge_index_i_s=image_simi['edge_index_i_s'],
    idx_to_id=text_simi['idx_to_id'],
    dialogue_items=dialogue_items,
    movielens_edges=ml_edges,
    num_ml_users=n_ml_users,
)
graph_info = builder.get_graph_info()
```

#### Thay đổi 3: Tạo UHCSGPrompt thay vì MMPrompt
```python
# TRƯỚC (MSCRS)
prompt_encoder = MMPrompt(
    model.config.n_embd, text_encoder.config.hidden_size,
    model.config.n_head, model.config.n_layer, 2,
    n_entity=kg['num_entities'], num_relations=kg['num_relations'],
    num_bases=args.num_bases,
    edge_index=kg['edge_index'], edge_type=kg['edge_type'],
    edge_index_c=co['edge_index_c'],
    edge_index_i_s=image_simi['edge_index_i_s'],
    edge_index_t_s=text_simi['edge_index_t_s'],
    idx_to_id=text_simi['idx_to_id'],
)

# SAU (UH-CSG)
prompt_encoder = UHCSGPrompt(
    hidden_size=model.config.n_embd,
    token_hidden_size=text_encoder.config.hidden_size,
    n_head=model.config.n_head,
    n_layer=model.config.n_layer,
    n_block=2,
    n_entity=kg['num_entities'],
    num_relations=kg['num_relations'],
    num_bases=args.num_bases,
    edge_index=kg['edge_index'],
    edge_type=kg['edge_type'],
    # UH-CSG specific:
    unified_edge_index=graph_info['unified_edge_index'],
    num_dialogues=graph_info['num_dialogues'],
    num_ml_users=graph_info['num_ml_users'],
    dialogue_item_map=graph_info['dialogue_item_map'],
    movielens_edges=graph_info.get('movielens_edges'),
    num_lightgcn_layers=4,
    # Nếu cần prefix:
    n_prefix_rec=args.n_prefix_rec,  # cho finetune rec
    # n_prefix_conv=args.n_prefix_conv,  # cho conv
)
```

#### Forward call: KHÔNG CẦN THAY ĐỔI
```python
# Cả MSCRS lẫn UH-CSG đều dùng cùng interface:
prompt_embeds, loss_cl, loss_lb, entity_embeds_all = prompt_encoder(
    entity_ids=batch['entity'],
    token_embeds=token_embeds,
    output_entity=True,
)
batch['context']['prompt_embeds'] = prompt_embeds
batch['context']['entity_embeds'] = entity_embeds_all
```

---

## 3. Áp dụng cho từng script

### 3.1 Pretrain ReDial (`train_pre_redial.py`)
→ Xem `train_pre_redial_uhcsg.py` (file mẫu đầy đủ)

### 3.2 Finetune Rec ReDial (`train_rec_redial.py`)
Thay đổi giống hệt pretrain, thêm:
- `n_prefix_rec=args.n_prefix_rec` khi tạo UHCSGPrompt
- Load pretrained weights: `prompt_encoder.load(args.prompt_encoder)`

### 3.3 Finetune Conv (`train_conv.py`)
Thay đổi giống hệt, thêm:
- `n_prefix_conv=args.n_prefix_conv` khi tạo UHCSGPrompt
- Nếu conv/src dùng MMPrompt khác (với prompt_max_length, n_examples), 
  thêm các params tương ứng vào UHCSGPrompt constructor

### 3.4 INSPIRED dataset
Thay đổi tương tự, chỉ khác:
- Dùng `MMPrompt_inspired` → `UHCSGPrompt` (cùng class, ko cần variant)
- Data loading từ inspired dataset (paths khác)

---

## 4. Xử lý MovieLens data thực

### Bước 1: Download MovieLens 25M
```bash
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip
```

### Bước 2: Tạo mapping MovieLens movieId → DBpedia entity ID

Cách 1 — **Dùng links.csv + TMDB API** (chính xác nhất):
```python
# MovieLens links.csv có: movieId, imdbId, tmdbId
# ReDial dataset có entity IDs từ DBpedia
# Bạn cần:
# 1. Từ tmdbId → lấy title + year
# 2. Match title+year với entity names trong entity2id.json
# 3. Lưu mapping: {ml_movieId: entity_id}
```

Cách 2 — **Title matching** (đơn giản hơn):
```python
import json
import csv

# Load entity2id
with open('data/redial/entity2id.json') as f:
    entity2id = json.load(f)

# Load MovieLens movies
ml_movies = {}
with open('ml-25m/movies.csv') as f:
    for row in csv.DictReader(f):
        ml_movies[int(row['movieId'])] = row['title']

# Simple title matching
mapping = {}
for ml_id, ml_title in ml_movies.items():
    # Strip year: "Toy Story (1995)" -> "Toy Story"
    clean = ml_title.rsplit('(', 1)[0].strip()
    for ename, eid in entity2id.items():
        if clean.lower() in ename.lower():
            mapping[ml_id] = eid
            break

# Save
with open('ml_to_entity.json', 'w') as f:
    json.dump(mapping, f)
```

### Bước 3: Build edges
```python
import json
import csv
import torch
from collections import defaultdict

# Load mapping
with open('ml_to_entity.json') as f:
    ml_to_entity = json.load(f)
    ml_to_entity = {int(k): v for k, v in ml_to_entity.items()}

# Load ratings (keep rating >= 3.5)
user_items = defaultdict(list)
with open('ml-25m/ratings.csv') as f:
    for row in csv.DictReader(f):
        if float(row['rating']) >= 3.5:
            uid = int(row['userId'])
            mid = int(row['movieId'])
            if mid in ml_to_entity:
                user_items[uid].append(ml_to_entity[mid])

# Filter users with >= 3 items
filtered = {uid: items for uid, items in user_items.items() if len(items) >= 3}

# Build edge tensor
src, dst = [], []
user_map = {}
for uid, items in filtered.items():
    local_uid = len(user_map)
    user_map[uid] = local_uid
    for eid in items:
        src.append(local_uid)
        dst.append(eid)

edges = torch.tensor([src, dst], dtype=torch.long)
torch.save({
    'edges': edges,
    'num_users': len(user_map),
}, 'movielens_edges_redial.pt')
print(f"Saved: {len(user_map)} users, {edges.shape[1]} edges")
```

### Bước 4: Sử dụng trong training
```bash
python train_pre_redial_uhcsg.py \
    --use_mock_data=False \
    --movielens_file movielens_edges_redial.pt
```

---

## 5. Xử lý dialogue items thực

Thay vì mock data, extract entities từ training dialogues:

```python
from uhcsg_graph_builder import extract_dialogue_items_from_dataset

# File training data phải có format:
# Mỗi dòng là JSON với fields: conv_id, entity (list of entity IDs)
dialogue_items = extract_dialogue_items_from_dataset(
    dataset_dir='data',
    dataset='redial',
    split='train'
)
```

Nếu format file khác, tùy chỉnh hàm `extract_dialogue_items_from_dataset` 
trong `uhcsg_graph_builder.py`.

---

## 6. Hyperparameter tuning

| Parameter | Mặc định | Gợi ý range | Ghi chú |
|-----------|----------|-------------|---------|
| `num_lightgcn_layers` | 4 | 2-6 | >4 dễ over-smooth |
| `mock_ml_users` | 100 | 50-500 | Chỉ cho mock, real data tự xác định |
| `top_k` (text/image sim) | 20 | 10-30 | Trong dataset_dbpedia.py, không đổi |
| `learning_rate` | 5e-4 | 1e-4 – 1e-3 | Giữ nguyên hoặc giảm nhẹ |

---

## 7. Ablation studies đề xuất

Để validate contribution của từng component:

```bash
# Full model
python train_pre_redial_uhcsg.py --num_lightgcn_layers 4

# Ablation 1: Không có MovieLens edges
python train_pre_redial_uhcsg.py --use_mock_data False --movielens_file None

# Ablation 2: Số layers
python train_pre_redial_uhcsg.py --num_lightgcn_layers 1
python train_pre_redial_uhcsg.py --num_lightgcn_layers 2
python train_pre_redial_uhcsg.py --num_lightgcn_layers 6

# Ablation 3: Seen vs unseen items (cần thêm evaluation code)
# → Chia test items thành 2 nhóm: có trong train vs không có
# → Evaluate Recall@k riêng cho mỗi nhóm
```