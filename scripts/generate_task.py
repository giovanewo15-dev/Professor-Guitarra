#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "data" / "tasks.json"


def load_tasks() -> Dict[str, Any]:
    with TASKS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_tasks(data: Dict[str, Any]) -> None:
    with TASKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_history(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    history = data.get("history")
    if isinstance(history, list):
        return history
    legacy = data.get("historical_topics", [])
    if isinstance(legacy, list):
        return legacy
    return []


def build_prompt(data: Dict[str, Any]) -> List[Dict[str, str]]:
    current = data.get("current_task", {})
    student_level = data.get("student_level", {})
    history = normalize_history(data)
    today = dt.datetime.now().astimezone().strftime("%Y-%m-%d")

    system = (
        "Você é um professor de guitarra e mentor pedagógico. "
        "Crie uma tarefa diária com dois blocos obrigatórios: técnico e teórico. "
        "A sequência deve ser estritamente lógica, sem pular fundamentos. "
        "Nunca repita tópicos já presentes no histórico. "
        "Se o histórico mostrar que o aluno ainda está no diagnóstico, comece fazendo perguntas objetivas para mapear técnica e teoria. "
        "Cada bloco precisa conter: explicação clara, exemplo aplicado à guitarra, exercício de fixação e resultado esperado. "
        "O bloco técnico também precisa indicar técnica trabalhada, posição das mãos, dedilhado ou palhetada, BPM inicial, progressão de velocidade e expectativa de uma semana. "
        "O bloco teórico deve usar o braço da guitarra como referência visual sempre que fizer sentido."
    )

    user = {
        "today": today,
        "current_task": current,
        "student_level": student_level,
        "history": history,
        "pedagogical_sequence": [
            "diagnóstico inicial",
            "intervalos",
            "escala maior",
            "tríades",
            "formação de acordes",
            "campo harmônico maior",
            "harmonização",
            "modos"
        ],
        "rules": {
            "no_repeat_topics": True,
            "do_not_advance_without_foundation": True,
            "maintain_continuity": True
        }
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def parse_task(raw: str) -> Dict[str, Any]:
    return json.loads(raw)


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY não configurada.")

    data = load_tasks()
    current = data.get("current_task", {})
    if current.get("status") != "studied":
        print("Tarefa atual ainda não foi marcada como estudada. Nenhuma nova tarefa gerada.")
        return

    client = OpenAI()
    messages = build_prompt(data)
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        messages=messages,
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("A API da OpenAI não retornou conteúdo.")

    task = parse_task(content)
    history = normalize_history(data)
    previous = current.copy()

    if previous:
        history.append(
            {
                "date": previous.get("generated_at"),
                "topic": previous.get("topic"),
                "title": previous.get("message") or previous.get("topic"),
                "status": previous.get("status"),
            }
        )

    task.setdefault("generated_at", dt.datetime.now().astimezone().strftime("%Y-%m-%d"))
    task.setdefault("status", "pending")
    task.setdefault("topic", "Nova tarefa")
    task.setdefault("message", "Tarefa diária de guitarra")

    data["current_task"] = task
    data["history"] = history
    data["historical_topics"] = [item.get("topic") for item in history if item.get("topic")]
    data["student_level"] = task.get("student_level", data.get("student_level", {}))
    data["next_pedagogical_step"] = task.get("next_pedagogical_step", data.get("next_pedagogical_step", ""))

    save_tasks(data)
    print("Nova tarefa gerada com sucesso.")


if __name__ == "__main__":
    main()
