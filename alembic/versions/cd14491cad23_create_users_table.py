"""create users table

Revision ID: cd14491cad23
Revises: 
Create Date: 2023-11-13 19:13:01.988868

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cd14491cad23'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'users',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('user_name', sa.VARCHAR(length=255), nullable=False),
        sa.Column('picture_path', sa.VARCHAR(
            length=255), nullable=False, unique=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.func.now())

    )
    pass


def downgrade() -> None:
    op.drop_table('users')
    pass
