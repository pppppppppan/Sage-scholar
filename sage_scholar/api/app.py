from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sage_scholar.base import setup_logging
from sage_scholar.infrastructure.container import ApplicationContainer

from .routes import queries, runs, library, evidence, audits, memory


def create_app(container: ApplicationContainer | None = None) -> FastAPI:
    application_container = container or ApplicationContainer()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(enable_display=False)
        app.state.container = application_container
        if hasattr(application_container.run_manager, "recover"):
            await application_container.run_manager.recover()
        yield
        await application_container.close()

    app = FastAPI(title="SAGE Scholar", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "sage-scholar"}

    app.include_router(queries.router)
    app.include_router(runs.router)
    app.include_router(library.router)
    app.include_router(evidence.router)
    app.include_router(audits.router)
    app.include_router(memory.router)
    return app


app = create_app()
