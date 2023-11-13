"""create access history table

Revision ID: fe718d1b1d11
Revises: cd14491cad23
Create Date: 2023-11-13 19:14:11.229312

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fe718d1b1d11'
down_revision: Union[str, None] = 'cd14491cad23'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'access_history',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('user_id', sa.Integer, nullable=True),
        sa.Column('is_unknown', sa.Boolean, default=True),
        sa.Column('unknown_picture_path', sa.VARCHAR(
            length=255), nullable=True),
        sa.Column('accessed_at', sa.DateTime(
            timezone=True), server_default=sa.func.now())
    )

    op.create_foreign_key(
        'fk_access_history_user_id',
        'access_history', 'users',
        ['user_id'], ['id']
    )
    pass


def downgrade() -> None:
    op.drop_table('access_history')
    pass
