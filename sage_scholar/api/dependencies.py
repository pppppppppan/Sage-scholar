from typing import Annotated
from fastapi import Depends, Request
from sage_scholar.infrastructure.container import ApplicationContainer


def get_container(request: Request) -> ApplicationContainer:
    return request.app.state.container


Container = Annotated[ApplicationContainer, Depends(get_container)]
