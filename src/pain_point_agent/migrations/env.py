from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from pain_point_agent.db import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

url = context.get_x_argument(as_dictionary=True).get("url") or config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    context.configure(url=url, target_metadata=Base.metadata, literal_binds=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = engine_from_config({"sqlalchemy.url": url}, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
