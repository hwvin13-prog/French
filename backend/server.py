Harika! Şimdi tam bir Fransızca öğrenme uygulaması oluşturacağım. Backend ve frontend dosyalarını birlikte yazacağım:
Action: file_editor create /app/backend/server.py --file-text "from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from bson import ObjectId
from emergentintegrations.llm.chat import LlmChat, UserMessage
import random

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix=\"/api\")

# Get Emergent LLM key
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

# Helper function to convert ObjectId to string
def serialize_doc(doc):
    if doc and \"_id\" in doc:
        doc[\"id\"] = str(doc[\"_id\"])
        del doc[\"_id\"]
    return doc

# ==================== MODELS ====================

class User(BaseModel):
    name: str
    level: str = \"A1\"  # A1, A2, B1, B2, C1
    daily_goal: int = 10  # words per day
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ReadingText(BaseModel):
    level: str
    title: str
    content: str
    words: List[Dict[str, str]]  # [{french: \"bonjour\", turkish: \"merhaba\"}]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Vocabulary(BaseModel):
    level: str
    french: str
    turkish: str
    example_sentence: Optional[str] = None
    category: Optional[str] = None

class UserVocabulary(BaseModel):
    user_id: str
    word_id: str
    french: str
    turkish: str
    added_at: datetime = Field(default_factory=datetime.utcnow)
    last_reviewed: Optional[datetime] = None
    ease_factor: float = 2.5  # For spaced repetition
    interval: int = 0  # Days until next review
    repetitions: int = 0
    correct_count: int = 0
    incorrect_count: int = 0

class ListeningText(BaseModel):
    level: str
    title: str
    text: str
    audio_base64: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class WritingPrompt(BaseModel):
    level: str
    topic: str
    prompt: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserWriting(BaseModel):
    user_id: str
    prompt_id: str
    prompt_text: str
    user_text: str
    ai_feedback: Optional[str] = None
    corrections: Optional[List[Dict[str, str]]] = None
    score: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChatMessage(BaseModel):
    user_id: str
    role: str  # \"user\" or \"assistant\"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# ==================== USER ENDPOINTS ====================

@api_router.post(\"/user\")
async def create_user(user: User):
    user_dict = user.model_dump()
    result = await db.users.insert_one(user_dict)
    user_dict[\"id\"] = str(result.inserted_id)
    return serialize_doc(user_dict)

@api_router.get(\"/user/{user_id}\")
async def get_user(user_id: str):
    user = await db.users.find_one({\"_id\": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail=\"User not found\")
    return serialize_doc(user)

@api_router.put(\"/user/{user_id}/level\")
async def update_user_level(user_id: str, level: str):
    result = await db.users.update_one(
        {\"_id\": ObjectId(user_id)},
        {\"$set\": {\"level\": level}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail=\"User not found\")
    return {\"message\": \"Level updated\", \"level\": level}

# ==================== READING ENDPOINTS ====================

@api_router.get(\"/reading\")
async def get_reading_texts(level: Optional[str] = None):
    query = {\"level\": level} if level else {}
    texts = await db.reading_texts.find(query).to_list(100)
    return [serialize_doc(text) for text in texts]

@api_router.get(\"/reading/{text_id}\")
async def get_reading_text(text_id: str):
    text = await db.reading_texts.find_one({\"_id\": ObjectId(text_id)})
    if not text:
        raise HTTPException(status_code=404, detail=\"Reading text not found\")
    return serialize_doc(text)

@api_router.post(\"/reading\")
async def create_reading_text(reading: ReadingText):
    reading_dict = reading.model_dump()
    result = await db.reading_texts.insert_one(reading_dict)
    reading_dict[\"id\"] = str(result.inserted_id)
    return serialize_doc(reading_dict)

# ==================== VOCABULARY ENDPOINTS ====================

@api_router.get(\"/vocabulary\")
async def get_vocabulary(level: Optional[str] = None):
    query = {\"level\": level} if level else {}
    words = await db.vocabulary.find(query).to_list(1000)
    return [serialize_doc(word) for word in words]

@api_router.post(\"/vocabulary\")
async def create_vocabulary(vocab: Vocabulary):
    vocab_dict = vocab.model_dump()
    result = await db.vocabulary.insert_one(vocab_dict)
    vocab_dict[\"id\"] = str(result.inserted_id)
    return serialize_doc(vocab_dict)

# ==================== USER VOCABULARY (BASKET) ENDPOINTS ====================

@api_router.get(\"/user-vocabulary/{user_id}\")
async def get_user_vocabulary(user_id: str):
    words = await db.user_vocabulary.find({\"user_id\": user_id}).to_list(1000)
    return [serialize_doc(word) for word in words]

@api_router.post(\"/user-vocabulary\")
async def add_to_vocabulary_basket(vocab: UserVocabulary):
    # Check if already exists
    existing = await db.user_vocabulary.find_one({
        \"user_id\": vocab.user_id,
        \"word_id\": vocab.word_id
    })
    if existing:
        return serialize_doc(existing)
    
    vocab_dict = vocab.model_dump()
    result = await db.user_vocabulary.insert_one(vocab_dict)
    vocab_dict[\"id\"] = str(result.inserted_id)
    return serialize_doc(vocab_dict)

@api_router.delete(\"/user-vocabulary/{vocab_id}\")
async def remove_from_vocabulary_basket(vocab_id: str):
    result = await db.user_vocabulary.delete_one({\"_id\": ObjectId(vocab_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail=\"Word not found in basket\")
    return {\"message\": \"Word removed from basket\"}

# ==================== PRACTICE/LEARNING ENDPOINTS ====================

@api_router.get(\"/practice/{user_id}/due\")
async def get_due_words(user_id: str):
    \"\"\"Get words due for review (spaced repetition)\"\"\"
    now = datetime.utcnow()
    words = await db.user_vocabulary.find({
        \"user_id\": user_id
    }).to_list(1000)
    
    due_words = []
    for word in words:
        if word.get(\"last_reviewed\"):
            next_review = word[\"last_reviewed\"] + timedelta(days=word.get(\"interval\", 0))
            if next_review <= now:
                due_words.append(serialize_doc(word))
        else:
            # Never reviewed, include it
            due_words.append(serialize_doc(word))
    
    return due_words

@api_router.post(\"/practice/review\")
async def review_word(vocab_id: str, correct: bool):
    \"\"\"Update spaced repetition data after review\"\"\"
    word = await db.user_vocabulary.find_one({\"_id\": ObjectId(vocab_id)})
    if not word:
        raise HTTPException(status_code=404, detail=\"Word not found\")
    
    ease_factor = word.get(\"ease_factor\", 2.5)
    interval = word.get(\"interval\", 0)
    repetitions = word.get(\"repetitions\", 0)
    
    if correct:
        if repetitions == 0:
            interval = 1
        elif repetitions == 1:
            interval = 3
        else:
            interval = int(interval * ease_factor)
        repetitions += 1
        ease_factor = max(1.3, ease_factor + 0.1)
        correct_count = word.get(\"correct_count\", 0) + 1
        incorrect_count = word.get(\"incorrect_count\", 0)
    else:
        interval = 0
        repetitions = 0
        ease_factor = max(1.3, ease_factor - 0.2)
        correct_count = word.get(\"correct_count\", 0)
        incorrect_count = word.get(\"incorrect_count\", 0) + 1
    
    await db.user_vocabulary.update_one(
        {\"_id\": ObjectId(vocab_id)},
        {
            \"$set\": {
                \"last_reviewed\": datetime.utcnow(),
                \"ease_factor\": ease_factor,
                \"interval\": interval,
                \"repetitions\": repetitions,
                \"correct_count\": correct_count,
                \"incorrect_count\": incorrect_count
            }
        }
    )
    
    return {\"message\": \"Review recorded\", \"next_review_days\": interval}

@api_router.get(\"/practice/{user_id}/quiz\")
async def get_quiz_questions(user_id: str, count: int = 10):
    \"\"\"Generate multiple choice quiz questions\"\"\"
    user_words = await db.user_vocabulary.find({\"user_id\": user_id}).to_list(100)
    if len(user_words) < 4:
        raise HTTPException(status_code=400, detail=\"Not enough words in basket\")
    
    questions = []
    selected_words = random.sample(user_words, min(count, len(user_words)))
    all_words = await db.user_vocabulary.find({\"user_id\": user_id}).to_list(1000)
    
    for word in selected_words:
        # Get 3 random wrong answers
        wrong_answers = random.sample([w for w in all_words if w[\"_id\"] != word[\"_id\"]], min(3, len(all_words) - 1))
        options = [word[\"turkish\"]] + [w[\"turkish\"] for w in wrong_answers]
        random.shuffle(options)
        
        questions.append({
            \"id\": str(word[\"_id\"]),
            \"question\": word[\"french\"],
            \"options\": options,
            \"correct_answer\": word[\"turkish\"]
        })
    
    return questions

# ==================== LISTENING ENDPOINTS ====================

@api_router.get(\"/listening\")
async def get_listening_texts(level: Optional[str] = None):
    query = {\"level\": level} if level else {}
    texts = await db.listening_texts.find(query).to_list(100)
    return [serialize_doc(text) for text in texts]

@api_router.post(\"/listening\")
async def create_listening_text(listening: ListeningText):
    listening_dict = listening.model_dump()
    result = await db.listening_texts.insert_one(listening_dict)
    listening_dict[\"id\"] = str(result.inserted_id)
    return serialize_doc(listening_dict)

# ==================== WRITING ENDPOINTS ====================

@api_router.get(\"/writing/prompt\")
async def get_daily_writing_prompt(level: str):
    \"\"\"Get a random writing prompt for the level\"\"\"
    prompts = await db.writing_prompts.find({\"level\": level}).to_list(100)
    if not prompts:
        # Create a default prompt if none exist
        default_prompt = {
            \"level\": level,
            \"topic\": \"Ma journée\",
            \"prompt\": \"Décrivez votre journée typique.\"
        }
        result = await db.writing_prompts.insert_one(default_prompt)
        default_prompt[\"_id\"] = result.inserted_id
        prompts = [default_prompt]
    
    selected = random.choice(prompts)
    return serialize_doc(selected)

@api_router.post(\"/writing/submit\")
async def submit_writing(writing: UserWriting):
    \"\"\"Submit writing and get AI feedback\"\"\"
    try:
        # Initialize LLM chat for feedback
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f\"writing_{writing.user_id}_{datetime.utcnow().timestamp()}\",
            system_message=f\"\"\"Sen bir Fransızca öğretmenisin. Öğrencinin yazısını kontrol et ve detaylı geri bildirim ver.
            
Öğrenci seviyesi: {writing.prompt_text}

Lütfen:
1. Hataları düzelt (gramer, yazım, kelime seçimi)
2. Daha iyi alternatifler öner
3. Genel değerlendirme yap
4. 100 üzerinden puan ver

Cevabını Türkçe ver.\"\"\"
        ).with_model(\"openai\", \"gpt-5.2\")
        
        user_message = UserMessage(
            text=f\"Öğrencinin yazdığı metin:\n\n{writing.user_text}\n\nBu metni kontrol edip detaylı geri bildirim ver.\"
        )
        
        feedback = await chat.send_message(user_message)
        
        # Save to database
        writing_dict = writing.model_dump()
        writing_dict[\"ai_feedback\"] = feedback
        result = await db.user_writings.insert_one(writing_dict)
        writing_dict[\"id\"] = str(result.inserted_id)
        
        return serialize_doc(writing_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f\"Error generating feedback: {str(e)}\")

@api_router.get(\"/writing/history/{user_id}\")
async def get_writing_history(user_id: str):
    writings = await db.user_writings.find({\"user_id\": user_id}).sort(\"created_at\", -1).to_list(50)
    return [serialize_doc(w) for w in writings]

# ==================== CHAT ENDPOINTS ====================

@api_router.post(\"/chat/message\")
async def send_chat_message(user_id: str, message: str, level: str):
    \"\"\"Chat with AI in French\"\"\"
    try:
        # Save user message
        user_msg = {
            \"user_id\": user_id,
            \"role\": \"user\",
            \"content\": message,
            \"timestamp\": datetime.utcnow()
        }
        await db.chat_messages.insert_one(user_msg)
        
        # Get recent chat history
        history = await db.chat_messages.find({\"user_id\": user_id}).sort(\"timestamp\", -1).limit(10).to_list(10)
        history.reverse()
        
        # Initialize LLM chat
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f\"chat_{user_id}\",
            system_message=f\"\"\"Sen bir Fransızca konuşma partnerisin. Kullanıcı seviyesi: {level}

Görevin:
1. Kullanıcıyla sadece Fransızca konuş
2. Seviyesine uygun kelimeler ve yapılar kullan
3. Doğal, akıcı sohbet et
4. Eğer kullanıcı hata yaparsa, nazikçe düzelt
5. Yeni kelimeler öğret
6. Konuşmayı canlı tut, sorular sor

Kullanıcı Türkçe yazarsa, \"Lütfen Fransızca konuşalım!\" de ve sorusunu Fransızca sor.\"\"\"
        ).with_model(\"openai\", \"gpt-5.2\")
        
        user_message = UserMessage(text=message)
        response = await chat.send_message(user_message)
        
        # Save assistant message
        assistant_msg = {
            \"user_id\": user_id,
            \"role\": \"assistant\",
            \"content\": response,
            \"timestamp\": datetime.utcnow()
        }
        await db.chat_messages.insert_one(assistant_msg)
        
        return {\"response\": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f\"Error in chat: {str(e)}\")

@api_router.get(\"/chat/history/{user_id}\")
async def get_chat_history(user_id: str, limit: int = 50):
    messages = await db.chat_messages.find({\"user_id\": user_id}).sort(\"timestamp\", -1).limit(limit).to_list(limit)
    messages.reverse()
    return [serialize_doc(msg) for msg in messages]

@api_router.delete(\"/chat/history/{user_id}\")
async def clear_chat_history(user_id: str):
    await db.chat_messages.delete_many({\"user_id\": user_id})
    return {\"message\": \"Chat history cleared\"}

# ==================== SEED DATA ====================

@api_router.post(\"/seed\")
async def seed_database():
    \"\"\"Seed database with sample content\"\"\"
    
    # Sample reading texts
    reading_texts = [
        {
            \"level\": \"A1\",
            \"title\": \"Bonjour!\",
            \"content\": \"Bonjour! Je m'appelle Marie. J'habite à Paris. J'aime le café et les croissants. Je suis étudiante.\",
            \"words\": [
                {\"french\": \"bonjour\", \"turkish\": \"merhaba\"},
                {\"french\": \"je m'appelle\", \"turkish\": \"benim adım\"},
                {\"french\": \"j'habite\", \"turkish\": \"yaşıyorum\"},
                {\"french\": \"j'aime\", \"turkish\": \"seviyorum\"},
                {\"french\": \"café\", \"turkish\": \"kahve\"},
                {\"french\": \"croissants\", \"turkish\": \"kruvasan\"},
                {\"french\": \"étudiante\", \"turkish\": \"öğrenci (kadın)\"}
            ]
        },
        {
            \"level\": \"A2\",
            \"title\": \"Ma Famille\",
            \"content\": \"Ma famille est grande. J'ai deux frères et une sœur. Mon père est médecin et ma mère est professeur. Nous habitons dans une grande maison.\",
            \"words\": [
                {\"french\": \"famille\", \"turkish\": \"aile\"},
                {\"french\": \"frères\", \"turkish\": \"erkek kardeşler\"},
                {\"french\": \"sœur\", \"turkish\": \"kız kardeş\"},
                {\"french\": \"père\", \"turkish\": \"baba\"},
                {\"french\": \"mère\", \"turkish\": \"anne\"},
                {\"french\": \"médecin\", \"turkish\": \"doktor\"},
                {\"french\": \"professeur\", \"turkish\": \"öğretmen\"},
                {\"french\": \"maison\", \"turkish\": \"ev\"}
            ]
        },
        {
            \"level\": \"B1\",
            \"title\": \"Les Vacances\",
            \"content\": \"L'été dernier, je suis allé en Provence. C'était magnifique! J'ai visité de nombreux villages pittoresques. La nourriture était délicieuse et les gens très accueillants.\",
            \"words\": [
                {\"french\": \"vacances\", \"turkish\": \"tatil\"},
                {\"french\": \"été\", \"turkish\": \"yaz\"},
                {\"french\": \"magnifique\", \"turkish\": \"muhteşem\"},
                {\"french\": \"villages\", \"turkish\": \"köyler\"},
                {\"french\": \"pittoresques\", \"turkish\": \"pitoresk\"},
                {\"french\": \"nourriture\", \"turkish\": \"yemek\"},
                {\"french\": \"délicieuse\", \"turkish\": \"lezzetli\"},
                {\"french\": \"accueillants\", \"turkish\": \"misafirperver\"}
            ]
        }
    ]
    
    # Sample vocabulary
    vocabulary = [
        {\"level\": \"A1\", \"french\": \"bonjour\", \"turkish\": \"merhaba\", \"example_sentence\": \"Bonjour, comment allez-vous?\"},
        {\"level\": \"A1\", \"french\": \"merci\", \"turkish\": \"teşekkürler\", \"example_sentence\": \"Merci beaucoup!\"},
        {\"level\": \"A1\", \"french\": \"au revoir\", \"turkish\": \"hoşçakal\", \"example_sentence\": \"Au revoir, à bientôt!\"},
        {\"level\": \"A1\", \"french\": \"oui\", \"turkish\": \"evet\", \"example_sentence\": \"Oui, c'est vrai.\"},
        {\"level\": \"A1\", \"french\": \"non\", \"turkish\": \"hayır\", \"example_sentence\": \"Non, ce n'est pas vrai.\"},
        {\"level\": \"A2\", \"french\": \"aujourd'hui\", \"turkish\": \"bugün\", \"example_sentence\": \"Aujourd'hui, il fait beau.\"},
        {\"level\": \"A2\", \"french\": \"demain\", \"turkish\": \"yarın\", \"example_sentence\": \"À demain!\"},
        {\"level\": \"A2\", \"french\": \"hier\", \"turkish\": \"dün\", \"example_sentence\": \"Hier, j'ai travaillé.\"},
        {\"level\": \"B1\", \"french\": \"cependant\", \"turkish\": \"ancak\", \"example_sentence\": \"Il est intelligent, cependant il est paresseux.\"},
        {\"level\": \"B1\", \"french\": \"néanmoins\", \"turkish\": \"yine de\", \"example_sentence\": \"C'est difficile, néanmoins c'est possible.\"}
    ]
    
    # Sample writing prompts
    writing_prompts = [
        {\"level\": \"A1\", \"topic\": \"Présentation\", \"prompt\": \"Présentez-vous: votre nom, où vous habitez, ce que vous aimez.\"},
        {\"level\": \"A2\", \"topic\": \"Ma Journée\", \"prompt\": \"Décrivez une journée typique de votre vie.\"},
        {\"level\": \"B1\", \"topic\": \"Voyage\", \"prompt\": \"Racontez un voyage mémorable que vous avez fait.\"},
        {\"level\": \"B2\", \"topic\": \"Opinion\", \"prompt\": \"Donnez votre opinion sur l'importance de l'apprentissage des langues.\"},
        {\"level\": \"C1\", \"topic\": \"Débat\", \"prompt\": \"Analysez les avantages et inconvénients de la technologie moderne.\"}
    ]
    
    # Clear existing data
    await db.reading_texts.delete_many({})
    await db.vocabulary.delete_many({})
    await db.writing_prompts.delete_many({})
    
    # Insert new data
    await db.reading_texts.insert_many(reading_texts)
    await db.vocabulary.insert_many(vocabulary)
    await db.writing_prompts.insert_many(writing_prompts)
    
    return {
        \"message\": \"Database seeded successfully\",
        \"reading_texts\": len(reading_texts),
        \"vocabulary\": len(vocabulary),
        \"writing_prompts\": len(writing_prompts)
    }

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[\"*\"],
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event(\"shutdown\")
async def shutdown_db_client():
    client.close()
"
Observation: Overwrite successful: /app/backend/server.py

Fetching debug logs for expo

==> /var/log/supervisor/expo.err.log <==
  expo-router@5.1.4 - expected version: ~6.0.21
  expo-splash-screen@0.30.10 - expected version: ~31.0.13
  expo-status-bar@2.2.3 - expected version: ~3.0.9
  expo-symbols@0.4.5 - expected version: ~1.0.8
  expo-system-ui@5.0.10 - expected version: ~6.0.9
  expo-web-browser@14.2.0 - expected version: ~15.0.10
  react@19.0.0 - expected version: 19.1.0
  react-dom@19.0.0 - expected version: 19.1.0
  react-native-gesture-handler@2.24.0 - expected version: ~2.28.0
  react-native-reanimated@3.17.5 - expected version: ~4.1.1
  react-native-safe-area-context@5.4.0 - expected version: ~5.6.0
  react-native-screens@4.11.1 - expected version: ~4.16.0
  react-native-web@0.20.0 - expected version: ^0.21.0
  react-native-webview@13.13.5 - expected version: 13.15.0
  @types/react@19.0.14 - expected version: ~19.1.10
  eslint-config-expo@9.2.0 - expected version: ~10.0.0
  typescript@5.8.3 - expected version: ~5.9.2
Your project may not work correctly until you install the expected versions of the packages.
λ  WARN  Image: style.resizeMode is deprecated. Please use props.resizeMode. 
λ  WARN  Image: style.resizeMode is deprecated. Please use props.resizeMode. 

==> /var/log/supervisor/expo.out.log <==
Web node_modules/expo-router/entry.js ░░░░░░░░░░░░░░░░  0.0% (0/1)
λ node_modules/expo-router/node/render.js ░░░░░░░░░░░░░░░░  0.0% (0/1)
λ Bundled 22520ms node_modules/expo-router/node/render.js (687 modules)
Web node_modules/expo-router/entry.js ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░ 99.9% (721/721)
λ node_modules/expo-router/node/render.js ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░ 99.9% (687/687)
Web node_modules/expo-router/entry.js ░░░░░░░░░░░░░░░░  0.0% (0/1)
λ node_modules/expo-router/node/render.js ░░░░░░░░░░░░░░░░  0.0% (0/1)
λ Bundled 21965ms node_modules/expo-router/node/render.js (1 module)
λ  LOG  https://frensircoach.preview.emergentagent.com EXPO_PUBLIC_BACKEND_URL
Web node_modules/expo-router/entry.js ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░ 99.9% (721/721)
Web node_modules/expo-router/entry.js ░░░░░░░░░░░░░░░░  0.0% (0/1)
λ  LOG  https://frensircoach.preview.emergentagent.com EXPO_PUBLIC_BACKEND_URL
Web node_modules/expo-router/entry.js ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░ 99.9% (721/721)
Web node_modules/expo-router/entry.js ░░░░░░░░░░░░░░░░  0.0% (0/1)
Web Bundled 28407ms node_modules/expo-router/entry.js (721 modules)
Web Bundled 27697ms node_modules/expo-router/entry.js (1 module)
Web node_modules/expo-router/entry.js ▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░ 91.9% (605/631)
Web Bundled 4300ms node_modules/expo-router/entry.js (722 modules)
 LOG  [web] Logs will appear in the browser console
 LOG  [web] Logs will appear in the browser console


Fetching debug logs for backend

==> /var/log/supervisor/backend.err.log <==
INFO:     Will watch for changes in these directories: ['/app/backend']
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Started reloader process [92] using WatchFiles
INFO:     Started server process [129]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [129]
INFO:     Stopping reloader process [92]
INFO:     Will watch for changes in these directories: ['/app/backend']
INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)
INFO:     Started reloader process [225] using WatchFiles
INFO:     Started server process [256]
INFO:     Waiting for application startup.
INFO:     Application startup complete.

==> /var/log/supervisor/backend.out.log <==


Fetching debug logs for mongodb

tail: cannot open '/var/log/supervisor/mongodb*.log' for reading: No such file or directory
