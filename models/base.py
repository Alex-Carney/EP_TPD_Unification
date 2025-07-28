"""
Nothing added here, just for inheritance. In the future if we want
to add a custom base class, such as adding created_at and updated_at,
we can do it here
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass