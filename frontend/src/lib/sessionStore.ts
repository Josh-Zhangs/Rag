import { openDB, type IDBPDatabase } from 'idb'
import type { Session, Turn } from '../types'

// ── 数据库配置 ──────────────────────────────────────────────

const DB_NAME = 'machiariRAG'
const DB_VERSION = 1
const STORE_NAME = 'sessions'

// 数据库连接 Promise（单例）
let dbPromise: Promise<IDBPDatabase> | null = null

// 获取数据库连接（单例）
function getDB(): Promise<IDBPDatabase> {
    if (!dbPromise) {
        dbPromise = openDB(DB_NAME, DB_VERSION, {
            upgrade(db) {
                // 首次创建或版本升级时创建 object store
                if (!db.objectStoreNames.contains(STORE_NAME)) {
                    db.createObjectStore(STORE_NAME, { keyPath: 'id' })
                }
            },
        })
    }
    return dbPromise
}

// ── 会话 CRUD ────────────────────────────────────────────────

// 创建新会话，返回会话 ID
export async function createSession(): Promise<string> {
    // 生成唯一 ID：时间戳 + 6 位随机后缀
    const id = `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
    const session: Session = {
        id,
        title: '新規会話',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        turns: [],
    }
    const db = await getDB()
    await db.put(STORE_NAME, session)
    return id
}

// 获取指定会话
export async function getSession(id: string): Promise<Session | undefined> {
    const db = await getDB()
    return db.get(STORE_NAME, id)
}

// 获取所有会话（按最后更新时间降序）
export async function getAllSessions(): Promise<Session[]> {
    const db = await getDB()
    const all = await db.getAll(STORE_NAME)
    return all.sort((a, b) => b.updatedAt - a.updatedAt)
}

// 追加一轮对话到指定会话
export async function appendTurn(id: string, turn: Turn): Promise<void> {
    const db = await getDB()
    const session = await db.get(STORE_NAME, id) as Session | undefined
    if (!session) {
        console.warn(`Session not found: ${id}`)
        return
    }
    // 追加新轮次
    session.turns.push(turn)
    // 第一轮时自动生成标题（问题截断 20 字）
    if (session.turns.length === 1) {
        session.title = turn.question.slice(0, 20)
    }
    session.updatedAt = Date.now()
    await db.put(STORE_NAME, session)
}

// 删除指定会话
export async function deleteSession(id: string): Promise<void> {
    const db = await getDB()
    await db.delete(STORE_NAME, id)
}

// ── 活跃会话指针 ─────────────────────────────────────────────

const ACTIVE_SESSION_KEY = 'machiariRAG_activeSessionId'

// 获取当前活跃会话 ID
export function getActiveSessionId(): string | null {
    return localStorage.getItem(ACTIVE_SESSION_KEY)
}

// 设置当前活跃会话 ID
export function setActiveSessionId(id: string): void {
    localStorage.setItem(ACTIVE_SESSION_KEY, id)
}

// 清除活跃会话 ID（删除会话时调用）
export function clearActiveSessionId(): void {
    localStorage.removeItem(ACTIVE_SESSION_KEY)
}
