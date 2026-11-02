from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column()         # mapped_column(...)
    content: Mapped[str] = mapped_column(Text)     # texto inteiro, tipo Text
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())  # server_default=func.now()

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))    # ForeignKey("documents.id", ondelete="CASCADE")
    position: Mapped[int] = mapped_column()       # ordem do trecho no documento
    text: Mapped[str] = mapped_column(Text)     # texto inteiro, tipo Text
    embedding: Mapped[list[float]] = mapped_column(Vector(384))  # Vector(384)

    document: Mapped["Document"] = relationship(back_populates="chunks")