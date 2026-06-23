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
    curriculum = data.get("curriculum", {})
    history = normalize_history(data)
    today = dt.datetime.now().astimezone().strftime("%Y-%m-%d")

    system = (
        "Você é um professor de guitarra e mentor pedagógico. "
        "Crie uma lição diária completa, não um diagnóstico. "
        "A lição deve ensinar com base conceitual consolidada, aplicação no braço da guitarra, exercício técnico, exercício teórico e revisão. "
        "A sequência deve respeitar a ordem pedagógica indicada no curriculum. "
        "Nunca repita tópicos já presentes no histórico. "
        "Cada lição precisa ensinar uma base clara, usar exemplos aplicados à guitarra e terminar com critérios de checagem do entendimento. "
        "O bloco técnico deve indicar técnica, posição das mãos, palhetada ou dedilhado, BPM inicial, progressão e resultado esperado em uma semana. "
        "O bloco teórico deve explicar o conceito, mostrar o mapa no braço da guitarra e propor fixação prática."
    )

    user = {
        "today": today,
        "current_task": current,
        "student_level": student_level,
        "curriculum": curriculum,
        "history": history,
        "lesson_requirements": {
            "must_have": [
                "objective",
                "theory",
                "guitar_map",
                "application",
                "technical",
                "review",
            ],
            "style": "ensino claro, aplicado, sem superficialidade",
        },
        "rules": {
            "no_repeat_topics": True,
            "do_not_advance_without_foundation": True,
            "maintain_continuity": True,
        },
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
    task.setdefault("type", "lesson")
    task.setdefault("topic", "Nova lição")
    task.setdefault("message", "Lição diária de guitarra")
    task.setdefault("lesson", {})
    task.setdefault("curriculum_step", data.get("curriculum", {}).get("core_order", [None])[0])

    data["current_task"] = task
    data["history"] = history
    data["historical_topics"] = [item.get("topic") for item in history if item.get("topic")]
    data["student_level"] = task.get("student_level", data.get("student_level", {}))
    data["next_pedagogical_step"] = task.get("next_pedagogical_step", data.get("next_pedagogical_step", ""))
    data["last_generation_mode"] = "lesson"

    save_tasks(data)
    print("Nova lição gerada com sucesso.")


if __name__ == "__main__":
    main()
