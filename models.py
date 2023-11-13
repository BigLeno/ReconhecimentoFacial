import datetime

from typing import List
from sqlalchemy import DateTime, ForeignKey, String, Boolean, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_name: Mapped[str] = mapped_column(String(255))
    picture_path: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
    accesses: Mapped[List["AccessHistory"]] = relationship()
    pass


class AccessHistory(Base):
    __tablename__ = "access_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped["User"] = relationship(back_populates="accesses")
    is_unknown: Mapped[bool] = mapped_column(Boolean)
    unknown_picture_path: Mapped[str] = mapped_column(String(255))
    accessed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
    pass
