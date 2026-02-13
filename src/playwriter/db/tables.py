from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DBSeed(Base):
    __tablename__ = "seeds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    teleology: Mapped[str] = mapped_column(Text, default="")
    context: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    characters: Mapped[list[DBSeedCharacter]] = relationship(
        back_populates="seed", cascade="all, delete-orphan"
    )
    threads: Mapped[list[DBSeedThread]] = relationship(
        back_populates="seed", cascade="all, delete-orphan"
    )


class DBSeedCharacter(Base):
    __tablename__ = "seed_characters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    seed_id: Mapped[int] = mapped_column(ForeignKey("seeds.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")

    seed: Mapped[DBSeed] = relationship(back_populates="characters")


class DBSeedThread(Base):
    __tablename__ = "seed_threads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    seed_id: Mapped[int] = mapped_column(ForeignKey("seeds.id"), nullable=False)
    thread: Mapped[str] = mapped_column(Text, nullable=False)

    seed: Mapped[DBSeed] = relationship(back_populates="threads")


class DBCharacter(Base):
    __tablename__ = "characters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    seed_id: Mapped[int] = mapped_column(ForeignKey("seeds.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    profile_json: Mapped[str] = mapped_column(Text, default="{}")

    def get_profile(self) -> dict:
        return json.loads(self.profile_json)


class DBScene(Base):
    __tablename__ = "scenes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    seed_id: Mapped[int] = mapped_column(ForeignKey("seeds.id"), nullable=True)
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    data_json: Mapped[str] = mapped_column(Text, default="{}")


class DBGameSession(Base):
    __tablename__ = "game_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    seed_id: Mapped[int] = mapped_column(ForeignKey("seeds.id"), nullable=True)
    state_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    messages: Mapped[list[DBGameMessage]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class DBGameMessage(Base):
    __tablename__ = "game_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("game_sessions.id"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    speaker: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped[DBGameSession] = relationship(back_populates="messages")
