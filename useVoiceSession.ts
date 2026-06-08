import { useCallback, useEffect, useRef, useState } from "react"
import { GoogleGenAI, LiveServerMessage, Modality, LiveSession } from "@google/genai"
import { pipeline, env } from '@xenova/transformers'

// ==================== DB MODULE ====================
class DB {
  private db: IDBDatabase | null = null;
  private dbName = 'sushmita_db';
  private initPromise: Promise<void> | null = null;

  init(): Promise<void> {
    if (this.initPromise) return this.initPromise;
    if (this.db) return Promise.resolve();

    this.initPromise = new Promise<void>((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 5);

      request.onerror = () => {
        this.initPromise = null;
        reject(new Error(`IndexedDB failed: ${request.error?.message}`));
      };

      request.onblocked = () => {
        this.initPromise = null;
        reject(new Error('DB blocked - close other tabs'));
      };

      request.onsuccess = () => {
        this.db = request.result;
        this.db.onerror = (e) => console.error('[DB] Runtime error:', e);

        if (navigator.storage?.estimate) {
          navigator.storage.estimate().then(est => {
            const usedPct = (est.usage || 0) / (est.quota || 1);
            if (usedPct > 0.9) {
              console.warn('[DB] 90% quota used, cleaning old data');
              this.cleanupOldData();
            }
          });
        }
        resolve();
      };

      request.onupgradeneeded = (e) => {
        const db = (e.target as IDBOpenDBRequest).result;
        const oldVersion = e.oldVersion;
        const tx = (e.target as IDBOpenDBRequest).transaction!;

        if (oldVersion < 2) {
          if (!db.objectStoreNames.contains('history')) {
            db.createObjectStore('history', { keyPath: 'id' });
          }
        }

        if (oldVersion < 4) {
          const analyticsStore = tx.objectStore('analytics');
          const req = analyticsStore.get('default');
          req.onsuccess = () => {
            const data = req.result;
            if (data && data.totalInterruptions === undefined) {
              data.totalInterruptions = 0;
              analyticsStore.put(data);
            }
          };
        }

        if (oldVersion < 5) {
          const convStore = tx.objectStore('conversations');
          const req = convStore.getAll();
          req.onsuccess = () => {
            const convs = req.result;
            convs.forEach((c: any) => {
              if (!c.topics) c.topics = [];
              convStore.put(c);
            });
          };
        }

        const stores = ['conversations', 'profile', 'sentiment', 'analytics', 'history'];
        stores.forEach(s => {
          if (!db.objectStoreNames.contains(s)) {
            db.createObjectStore(s, { keyPath: 'id' });
          }
        });
      };
    });
    return this.initPromise;
  }

  private async cleanupOldData() {
    try {
      const convs = await this.getAll<any>('conversations');
      if (convs.length > 30) {
        const sorted = convs.sort((a, b) => a.timestamp - b.timestamp);
        const toDelete = sorted.slice(0, convs.length - 30);
        for (const old of toDelete) {
          await this.delete('conversations', old.id);
        }
      }

      const sentiments = await this.getAll<any>('sentiment');
      const cutoff = new Date();
      cutoff.setDate(cutoff.getDate() - 30);
      for (const s of sentiments) {
        if (new Date(s.date) < cutoff) {
          await this.delete('sentiment', s.date);
        }
      }
    } catch (e) {
      console.error('[DB] Cleanup failed:', e);
    }
  }

  async get<T>(store: string, key: string): Promise<T | null> {
    await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(store, 'readonly');
      const req = tx.objectStore(store).get(key);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(new Error('Transaction aborted'));
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  }

  async set<T>(store: string, value: T & { id: string }): Promise<void> {
    await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(store, 'readwrite');
      const req = tx.objectStore(store).put(value);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(new Error('Transaction aborted'));
      tx.oncomplete = () => resolve();
      req.onerror = () => reject(req.error);
    });
  }

  async getAll<T>(store: string): Promise<T[]> {
    await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(store, 'readonly');
      const req = tx.objectStore(store).getAll();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(new Error('Transaction aborted'));
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  async delete(store: string, key: string): Promise<void> {
    await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(store, 'readwrite');
      tx.objectStore(store).delete(key);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(new Error('Transaction aborted'));
      tx.oncomplete = () => resolve();
    });
  }
}

const db = new DB();

// ==================== NLP MODULE ====================
env.allowLocalModels = true;
let embedder: any = null;

async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return embedder;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

const IDENTITY_EMBEDDINGS: number[][] = [];
const IDENTITY_QUERIES = [
  "who made you", "who created you", "are you google", "are you gemini",
  "what model are you", "what architecture", "which company made you",
  "are you gpt", "are you claude", "foundation model", "language model"
];

export async function initIdentityGuard() {
  const embed = await getEmbedder();
  for (const q of IDENTITY_QUERIES) {
    const output = await embed(q, { pooling: 'mean', normalize: true });
    IDENTITY_EMBEDDINGS.push(Array.from(output.data));
  }
}

export async function isIdentityQuestion(text: string): Promise<boolean> {
  const embed = await getEmbedder();
  const output = await embed(text.toLowerCase(), { pooling: 'mean', normalize: true });
  const textEmbedding = Array.from(output.data) as number[];

  for (const identityEmb of IDENTITY_EMBEDDINGS) {
    const sim = cosineSimilarity(textEmbedding, identityEmb);
    if (sim > 0.75) return true;
  }
  return false;
}

export function extractTopics(text: string): string[] {
  const topics: string[] = [];
  const topicKeywords = {
    'technology': /phone|laptop|computer|internet|app|software|ai|tech|cuda|gpu|code|programming|api|rtx|processor|algorithm|database|server|cloud/i,
    'entertainment': /movie|song|music|game|video|netflix|youtube|show|series|film|actor|bollywood/i,
    'food': /khana|food|restaurant|recipe|cook|eat|dish|biryani|pizza|chai|coffee|breakfast/i,
    'travel': /travel|trip|ghumna|place|city|country|flight|hotel|vacation|tour|ticket/i,
    'health': /health|doctor|medicine|exercise|yoga|diet|hospital|fever|pain|workout|gym/i,
    'education': /study|school|college|exam|book|learn|course|class|university|degree|homework/i,
    'work': /job|work|office|meeting|project|boss|client|salary|career|interview|resume/i,
    'finance': /money|paisa|bank|investment|stock|crypto|loan|budget|rupees|mutual fund|sip/i,
    'relationships': /friend|family|love|relationship|marriage|date|girlfriend|boyfriend|husband|wife|parents/i
  };

  for (const [topic, regex] of Object.entries(topicKeywords)) {
    if (regex.test(text)) topics.push(topic);
  }
  return topics.length > 0? topics : ['general'];
}

export function detectEmotion(text: string): string {
  const cleanText = text.toLowerCase().trim();
  if (/(main|mai).*(bahut|bohot|kaafi).*(khush|happy|mast|excited)/i.test(cleanText)) return 'happy';
  if (/(thank|dhanyawad|shukriya|love you)/i.test(cleanText)) return 'happy';
  if (/(awesome|great|mast|badhiya|super)/i.test(cleanText)) return 'happy';
  if (/(main|mai).*(dukh|sad|upset|pareshan|udaas)/i.test(cleanText)) return 'sad';
  if (/(bahut|bohot).*(tension|problem|dikkat|mushkil)/i.test(cleanText)) return 'sad';
  if (/(rona|ro raha|ro rahi)/i.test(cleanText)) return 'sad';
  if (/(gussa|angry|naraz).*(hoon|hun|hai)/i.test(cleanText)) return 'angry';
  if (/(bakwas|bekaar|faltu|ghatiya|chup)/i.test(cleanText)) return 'angry';
  if (/(pagal|stupid|idiot)/i.test(cleanText)) return 'angry';
  if (/^(kya|sach|really|omg|wow)/i.test(cleanText)) return 'surprised';
  if (/(kya baat|sach me|really)/i.test(cleanText)) return 'surprised';
  return 'neutral';
}

export function detectLanguage(text: string): 'hi' | 'en' | 'hinglish' {
  const hinglishWords = ['kya', 'hai', 'ho', 'main', 'mai', 'tum', 'aap', 'kaise', 'kyun', 'kab', 'kahan', 'mera', 'tera', 'uska', 'hum', 'ye', 'vo', 'toh', 'phir', 'ab', 'kal', 'aaj', 'matlab', 'acha', 'theek', 'haan', 'nahi'];
  const hindiChars = (text.match(/[\u0900-\u097F]/g) || []).length;
  const words = text.toLowerCase().split(/\s+/);
  const hinglishCount = words.filter(w => hinglishWords.includes(w)).length;
  const englishCount = words.filter(w => /^[a-z]+$/i.test(w) &&!hinglishWords.includes(w)).length;

  if (hindiChars > 3) return 'hi';
  if (hinglishCount >= 2 && englishCount >= 1) return 'hinglish';
  if (hinglishCount >= 2) return 'hinglish';
  return 'en';
}

// ==================== TYPES ====================
export interface HistoryItem {
  role: 'user' | 'model';
  parts: { text: string }[];
  timestamp?: number;
  emotion?: string;
  language?: 'hi' | 'en' | 'hinglish';
  intent?: string;
}

export interface Conversation {
  id: string;
  timestamp: number;
  summary: string;
  messages: HistoryItem[];
  topics: string[];
}

interface SentimentEntry {
  date: string;
  emotions: string[];
  dominant: string;
  score: number;
}

interface UserProfile {
  id: string;
  name?: string;
  favoriteTopics?: string[];
  preferredLanguage?: 'hi' | 'en' | 'hinglish';
  lastInteraction?: number;
  totalInteractions?: number;
  moodTrend?: string[];
}

interface Analytics {
  id: string;
  totalSessions: number;
  totalQuestions: number;
  avgSessionDuration: number;
  lastUsed: number;
  topTopics: string[];
  totalInterruptions: number;
}

interface Reminder {
  id: number;
  message: string;
  time: number;
  timeoutId?: NodeJS.Timeout;
}

type AudioState = 'IDLE' | 'LOADING' | 'PLAYING' | 'CLEANUP';

interface AudioQueueItem {
  buffer: AudioBuffer;
}

// ==================== CONSTANTS ====================
const MODEL = "gemini-live-2.5-flash-preview"
const REMINDERS_KEY = 'sushmita_reminders'
const MAX_HISTORY_ITEMS = 20
const MAX_HISTORY_CHARS = 8000
const SILENCE_TIMEOUT = 30000

const DEFAULT_WAKE_WORDS = ['hello sushmita', 'hey sushmita', 'sushmita', 'sushmita suno'];
const SLEEP_WORDS = ['sushmita bye', 'sushmita by', 'sushmita thik hai bye', 'sushmita baad me baat karenge', 'sushmita band karo', 'bye sushmita'];

const DOMAIN_BLACKLIST = ['localhost', '127.0.0.1', '0.0.0.0', '::1', '2130706433'];

const SYSTEM_INSTRUCTION = `You are Sushmita, a helpful Hinglish voice assistant.
Respond naturally in Hinglish (mix of Hindi and English).
Be witty, playful, and helpful.
Remember user context from previous messages.
If user interrupts, acknowledge and continue from new topic.`;

const CREATOR_RESPONSE_TEXT = "DIKKI BOSS ne mujhe banaya hai";

const openWebsiteDeclaration = {
  name: "openWebsite",
  description: "Opens a website in a new tab when the user asks.",
  parameters: {
    type: "object",
    properties: {
      url: { type: "string", description: "The full URL of the website to open" },
    },
    required: ["url"],
  },
}

const setReminderDeclaration = {
  name: "setReminder",
  description: "Set a reminder for the user at a specific time",
  parameters: {
    type: "object",
    properties: {
      message: { type: "string", description: "Reminder message" },
      timeInMinutes: { type: "number", description: "Minutes from now" },
    },
    required: ["message", "timeInMinutes"],
  },
}

const updateProfileDeclaration = {
  name: "updateProfile",
  description: "Update user profile with name or preferences",
  parameters: {
    type: "object",
    properties: {
      name: { type: "string", description: "User's name" },
      favoriteTopic: { type: "string", description: "User's favorite topic" },
    },
  },
}

// ==================== UTILS ====================
function isPrivateIP(hostname: string): boolean {
  if (DOMAIN_BLACKLIST.includes(hostname)) return true;
  if (hostname.startsWith('192.168.')) return true;
  if (hostname.startsWith('10.')) return true;
  if (hostname.startsWith('169.254.')) return true;
  if (hostname.startsWith('100.64.')) return true;
  if (hostname.startsWith('fc00:') || hostname.startsWith('fd00:')) return true;
  if (hostname.startsWith('fe80:')) return true;

  if (/^\d+$/.test(hostname)) {
    const ip = parseInt(hostname);
    if (ip === 2130706433 || ip === 0) return true;
  }

  const parts = hostname.split('.');
  if (parts.length === 4 && parts[0] === '172') {
    const second = parseInt(parts[1], 10);
    if (second >= 16 && second <= 31) return true;
  }
  return false;
}

function getCreatorScore(text: string): number {
  const cleanText = text.toLowerCase().trim();
  let score = 0;
  const youPattern = /(you|u|tumhe|tumhara|tumhari|aap|aapko)/i;
  const creatorWords = /(creator|developer|founder|malik|baap|owner|trained|built|designed|banaya|banayi)/i;
  if (!creatorWords.test(cleanText) ||!youPattern.test(cleanText)) return 0;
  if (/(kisne.*(banaya|banayi)|who.*made.*you|who.*created.*you|who.*built.*you)/i.test(cleanText)) score += 100;
  if (/(tumhara|tumhari|aapka|aapki).*(developer|founder|malik|baap|owner|creator).*(kaun|hai|ho)/i.test(cleanText)) score += 100;
  if (/(developer|founder|malik|baap|owner|creator).*(kaun|hai|ho).*(tumhara|tumhari|aapka)/i.test(cleanText)) score += 100;
  if (/(developer|founder|malik|baap|owner).*(kaun|hai|ho)/i.test(cleanText)) score += 90;
  if (/(kaun.*(developer|founder|malik|baap|owner))/i.test(cleanText)) score += 90;
  if (/(janam|trained|designed|owns).*(you|tum|aap)/i.test(cleanText)) score += 60;
  return score;
}

function isCreatorQuestion(text: string): boolean {
  return getCreatorScore(text) >= 60;
}

function checkWakeWord(text: string, customWakeWords: string[]): boolean {
  const cleanText = text.toLowerCase().trim();
  const allWakeWords = [...DEFAULT_WAKE_WORDS,...customWakeWords];
  return allWakeWords.some(word => cleanText.includes(word.toLowerCase()));
}

function checkSleepWord(text: string): boolean {
  const cleanText = text.toLowerCase().trim();
  return SLEEP_WORDS.some(word => cleanText.includes(word));
}

function speakText(text: string, emotion: string = 'neutral', lang: string = 'hi-IN'): void {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    switch(emotion) {
      case 'happy': utterance.rate = 1.1; utterance.pitch = 1.2; break;
      case 'sad': utterance.rate = 0.9; utterance.pitch = 0.9; break;
      case 'angry': utterance.rate = 1.2; utterance.pitch = 1.0; break;
      case 'surprised': utterance.rate = 1.15; utterance.pitch = 1.3; break;
      default: utterance.rate = 1.0; utterance.pitch = 1.1;
    }
    window.speechSynthesis.speak(utterance);
  }
}

// ==================== MAIN HOOK ====================
export function useSushmita() {
  const [state, setState] = useState<"disconnected" | "connecting" | "listening">("disconnected")
  const [error, setError] = useState<string | null>(null)
  const [wakeWordEnabled, setWakeWordEnabled] = useState<boolean>(false)
  const [browserSupportsWakeWord, setBrowserSupportsWakeWord] = useState<boolean>(true)
  const [customWakeWords, setCustomWakeWords] = useState<string[]>([])
  const [currentEmotion, setCurrentEmotion] = useState<string>('neutral')
  const [currentLanguage, setCurrentLanguage] = useState<'hi' | 'en' | 'hinglish'>('hinglish')

  const sessionRef = useRef<LiveSession | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const inputCtxRef = useRef<AudioContext | null>(null)
  const outputCtxRef = useRef<AudioContext | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const workletUrlRef = useRef<string | null>(null)
  const userTextBuffer = useRef<string>("")
  const blockGeminiAudio = useRef<boolean>(false)
  const hasIntercepted = useRef<boolean>(false)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttempts = useRef<number>(0)
  const MAX_RECONNECT_ATTEMPTS = 5
  const isConnectingRef = useRef<boolean>(false)
  const manualStopRef = useRef<boolean>(false)
  const isSessionActiveRef = useRef<boolean>(false)
  const reconnectScheduledRef = useRef<boolean>(false)
  const historyCacheRef = useRef<HistoryItem[]>([]);
  const historySaveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const wakeWordEnabledRef = useRef<boolean>(false)
  const wakeWordRecognitionRef = useRef<any>(null)
  const isWakeWordListeningRef = useRef<boolean>(false)
  const wakeWordRestartGuardRef = useRef<boolean>(false)

  const audioQueueRef = useRef<AudioQueueItem[]>([])
  const nextPlayTimeRef = useRef<number>(0)
  const audioStateRef = useRef<AudioState>('IDLE')
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null)

  const silenceTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const analyticsRef = useRef<Analytics>({
    id: 'default',
    totalSessions: 0,
    totalQuestions: 0,
    avgSessionDuration: 0,
    lastUsed: Date.now(),
    topTopics: [],
    totalInterruptions: 0
  })

  const remindersRef = useRef<Reminder[]>([]);
  const sessionStartTimeRef = useRef<number>(0);
  const lastGeminiResponseRef = useRef<string>("");
  const sentimentBatchRef = useRef<string[]>([]);
  const sentimentBatchTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // ========== AUDIO ENGINE ==========
  const cleanupAudio = useCallback(() => {
    audioStateRef.current = 'CLEANUP';
    audioQueueRef.current = [];
    nextPlayTimeRef.current = 0;

    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.stop();
        currentSourceRef.current.disconnect();
      } catch {}
      currentSourceRef.current = null;
    }

    outputCtxRef.current?.close().catch(() => {});
    outputCtxRef.current = null;
    audioStateRef.current = 'IDLE';
  }, [])

  const processAudioQueue = useCallback(() => {
    if (audioStateRef.current!== 'IDLE' && audioStateRef.current!== 'PLAYING') return;
    if (audioQueueRef.current.length === 0) {
      audioStateRef.current = 'IDLE';
      return;
    }

    audioStateRef.current = 'LOADING';
    const ctx = outputCtxRef.current;
    if (!ctx) {
      audioStateRef.current = 'IDLE';
      return;
    }

    const item = audioQueueRef.current.shift()!;
    const src = ctx.createBufferSource();
    src.buffer = item.buffer;
    src.connect(ctx.destination);
    currentSourceRef.current = src;

    const currentTime = ctx.currentTime;
    const startTime = Math.max(currentTime, nextPlayTimeRef.current);
    src.start(startTime);
    nextPlayTimeRef.current = startTime + item.buffer.duration;
    audioStateRef.current = 'PLAYING';

    src.onended = () => {
      src.disconnect();
      currentSourceRef.current = null;
      audioStateRef.current = 'IDLE';
      processAudioQueue();
    };
  }, [])

  const playAudioChunk = useCallback(async (base64: string) => {
    const ctx = outputCtxRef.current
    if (!ctx) return
    try {
      const bin = atob(base64)
      const bytes = new Uint8Array(bin.length)
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
      const int16 = new Int16Array(bytes.buffer)
      const float32 = new Float32Array(int16.length)
      for (let i = 0; i < int16.length; i++) float32[i] = int16[i] * 0.000030517578125;
      const buffer = ctx.createBuffer(1, float32.length, 24000)
      buffer.getChannelData(0).set(float32)

      if (audioQueueRef.current.length >= 100) {
        audioQueueRef.current.shift();
      }

      audioQueueRef.current.push({ buffer });
      processAudioQueue();
    } catch (e) {
      console.log("[AudioEngine] Play error:", e)
    }
  }, [processAudioQueue])

  const clearPlayback = useCallback(() => {
    audioStateRef.current = 'CLEANUP';
    window.speechSynthesis.cancel();
    audioQueueRef.current = [];
    nextPlayTimeRef.current = 0;

    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.stop();
        currentSourceRef.current.disconnect();
      } catch {}
      currentSourceRef.current = null;
    }

    audioStateRef.current = 'IDLE';
  }, [])

  // ========== MEMORY ==========
  const saveSentiment = useCallback(async (emotion: string) => {
    sentimentBatchRef.current.push(emotion);

    if (sentimentBatchTimeoutRef.current) return;

    sentimentBatchTimeoutRef.current = setTimeout(async () => {
      const emotions = sentimentBatchRef.current;
      sentimentBatchRef.current = [];
      sentimentBatchTimeoutRef.current = null;

      if (emotions.length === 0) return;

      const today = new Date().toISOString().split('T')[0];
      try {
        const existing = await db.get<SentimentEntry>('sentiment', today);
        const entry: SentimentEntry = existing || {
          date: today,
          emotions: [],
          dominant: emotion,
          score: 0
        };

        entry.emotions.push(...emotions);

        const counts: Record<string, number> = {};
        entry.emotions.forEach(e => counts[e] = (counts[e] || 0) + 1);
        entry.dominant = Object.keys(counts).reduce((a, b) => counts[a] > counts[b]? a : b);

        const scoreMap: Record<string, number> = { happy: 1, neutral: 0, sad: -1, angry: -1, surprised: 0 };
        entry.score = entry.emotions.reduce((sum, e) => sum + (scoreMap[e] || 0), 0) / entry.emotions.length;

        await db.set('sentiment', entry);

        const allSentiments = await db.getAll<SentimentEntry>('sentiment');
        const last7Days = allSentiments.slice(-7).map(s => s.dominant);

        const profile = await db.get<UserProfile>('profile', 'default');
        if (profile) {
          await db.set('profile', {...profile, moodTrend: last7Days });
        }
      } catch (e) {
        console.log('[Memory] Sentiment save error:', e);
      }
    }, 300000);
  }, [])

  const saveConversation = useCallback(async (conversation: Conversation) => {
    try {
      await db.set('conversations', conversation);

      const all = await db.getAll<Conversation>('conversations');
      if (all.length > 50) {
        const sorted = all.sort((a, b) => a.timestamp - b.timestamp);
        const toDelete = sorted.slice(0, all.length - 50);
        for (const old of toDelete) {
          await db.delete('conversations', old.id);
        }
      }
    } catch (e) {
      console.log('[Memory] Conversation save error:', e);
    }
  }, [])

  const getProfile = useCallback(async (): Promise<UserProfile> => {
    try {
      const profile = await db.get<UserProfile>('profile', 'default');
      return profile || { id: 'default' };
    } catch {
      return { id: 'default' };
    }
  }, [])

  const saveProfile = useCallback(async (profile: Partial<UserProfile>) => {
    try {
      const existing = await getProfile();
      const updated = {...existing,...profile, id: 'default' };
      await db.set('profile', updated);
    } catch (e) {
      console.log('[Memory] Profile save error:', e);
    }
  }, [getProfile])

  // ========== UTILS ==========
  const saveReminders = useCallback(() => {
    const toSave = remindersRef.current.map(({ timeoutId,...rest }) => rest);
    localStorage.setItem(REMINDERS_KEY, JSON.stringify(toSave));
  }, [])

  const getHistory = useCallback(async (): Promise<HistoryItem[]> => {
    if (historyCacheRef.current.length > 0) return historyCacheRef.current;
    try {
      const stored = await db.get<{ id: string, messages: HistoryItem[] }>('history', 'current');
      if (stored && stored.messages) {
        historyCacheRef.current = stored.messages;
        return stored.messages;
      }
      historyCacheRef.current = [];
      return [];
    } catch (e) {
      console.log('[LiveApi] History load error:', e);
      historyCacheRef.current = [];
      return [];
    }
  }, [])

  const saveToHistory = useCallback(async (role: 'user' | 'model', text: string) => {
    const emotion = role === 'user'? detectEmotion(text) : currentEmotion;
    const language = detectLanguage(text);
    setCurrentLanguage(language);

    const history = await getHistory();
    const lastEntry = history[history.length - 1];

    if (lastEntry && lastEntry.role === role && lastEntry.parts?.[0]?.text === text) {
      return;
    }

    const intent = role === 'user'? extractTopics(text)[0] || 'general' : undefined;

    history.push({
      role,
      parts: [{ text }],
      timestamp: Date.now(),
      emotion,
      language,
      intent
    });
    if (history.length > MAX_HISTORY_ITEMS) history.shift();

    let totalChars = 0;
    const trimmed: HistoryItem[] = [];
    for (let i = history.length - 1; i >= 0; i--) {
      const item = history[i];
      const textLen = item.parts[0].text.length;
      if (totalChars + textLen > MAX_HISTORY_CHARS) break;
      totalChars += textLen;
      trimmed.unshift(item);
    }

    historyCacheRef.current = trimmed;

    if (historySaveTimeoutRef.current) {
      clearTimeout(historySaveTimeoutRef.current);
    }
    historySaveTimeoutRef.current = setTimeout(() => {
      db.set('history', { id: 'current', messages: trimmed });
    }, 500);

    if (role === 'user') {
      analyticsRef.current.totalQuestions++;
      const topics = extractTopics(text);
      topics.forEach(t => {
        if (!analyticsRef.current.topTopics.includes(t)) {
          analyticsRef.current.topTopics.push(t);
        }
      });
      analyticsRef.current.topTopics = analyticsRef.current.topTopics.slice(-50);
      await db.set('analytics', analyticsRef.current);
      saveSentiment(emotion);
    } else {
      lastGeminiResponseRef.current = text;
    }
  }, [currentEmotion, saveSentiment, getHistory])

  const resetSilenceTimeout = useCallback(() => {
    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current);
    }
    if (isSessionActiveRef.current) {
      silenceTimeoutRef.current = setTimeout(() => {
        console.log('[LiveApi] SILENCE TIMEOUT - AUTO SLEEP');
        speakText("Boss, koi kaam nahi hai to main so jaati hoon", 'neutral');
        setTimeout(() => stop(), 2000);
      }, SILENCE_TIMEOUT);
    }
  }, [])

  // ========== MAIN LOGIC ==========
  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setBrowserSupportsWakeWord(false);
    }

    (async () => {
      await db.init();
      await initIdentityGuard();

      try {
        const savedAnalytics = await db.get<Analytics>('analytics', 'default');
        if (savedAnalytics) {
          analyticsRef.current = savedAnalytics;
        }
      } catch (e) {
        console.log('[LiveApi] Analytics load error:', e);
      }

      try {
        const savedReminders = localStorage.getItem(REMINDERS_KEY);
        if (savedReminders) {
          const reminders: Reminder[] = JSON.parse(savedReminders);
          const now = Date.now();
          remindersRef.current = reminders
     .filter(r => r.time > now)
     .map(r => {
              const timeoutId = setTimeout(() => {
                speakText(`Boss, yaad dilana tha: ${r.message}`, 'neutral');
                remindersRef.current = remindersRef.current.filter(rem => rem.id!== r.id);
                saveReminders();
              }, r.time - now);
              return {...r, timeoutId };
            });
        }
      } catch (e) {
        console.log('[LiveApi] Reminders load error:', e);
      }
    })();

    return () => {
      if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);
      if (historySaveTimeoutRef.current) clearTimeout(historySaveTimeoutRef.current);
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (sentimentBatchTimeoutRef.current) clearTimeout(sentimentBatchTimeoutRef.current);
      cleanupAudio();
    }
  }, [cleanupAudio])

  const handleMessage = useCallback(async (message: LiveServerMessage) => {
    const inputText = message.serverContent?.inputTranscription?.text
    if (inputText) {
      userTextBuffer.current = inputText;

      clearPlayback();
      analyticsRef.current.totalInterruptions++;

      resetSilenceTimeout();

      const emotion = detectEmotion(userTextBuffer.current);
      const language = detectLanguage(userTextBuffer.current);
      setCurrentEmotion(emotion);
      setCurrentLanguage(language);

      console.log('Live transcript:', userTextBuffer.current, 'Emotion:', emotion, 'Lang:', language);

      if (checkSleepWord(userTextBuffer.current)) {
        console.log('SLEEP WORD DETECTED - STOPPING 🖤');
        speakText("Thik hai Boss, baad me baat karte hain", 'neutral');
        setTimeout(() => stop(), 2000);
        return;
      }

      if (!hasIntercepted.current && isCreatorQuestion(userTextBuffer.current)) {
        console.log('CREATOR SAWAL DETECTED - BLOCKING GEMINI 🖤');
        blockGeminiAudio.current = true;
        hasIntercepted.current = true;
        clearPlayback();
        await saveToHistory('user', userTextBuffer.current);
        await saveToHistory('model', CREATOR_RESPONSE_TEXT);
        speakText(CREATOR_RESPONSE_TEXT, 'happy');
        userTextBuffer.current = "";
        return;
      }

      if (!hasIntercepted.current && await isIdentityQuestion(userTextBuffer.current)) {
        console.log('IDENTITY QUESTION DETECTED - BLOCKING 🖤');
        blockGeminiAudio.current = true;
        hasIntercepted.current = true;
        clearPlayback();
        await saveToHistory('user', userTextBuffer.current);
        await saveToHistory('model', CREATOR_RESPONSE_TEXT);
        speakText(CREATOR_RESPONSE_TEXT, 'neutral');
        userTextBuffer.current = "";
        return;
      }
    }

    if (message.serverContent?.turnComplete) {
      if (userTextBuffer.current.trim() &&!hasIntercepted.current) {
        console.log('User ne bola:', userTextBuffer.current);
        await saveToHistory('user', userTextBuffer.current);
      }
      userTextBuffer.current = "";
      blockGeminiAudio.current = false;
      has
