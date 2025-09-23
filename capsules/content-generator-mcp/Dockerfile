FROM alquimiaai/base:latest

COPY --chown=1001:0 . .

RUN uv python install && uv sync --frozen --no-cache

USER 0
RUN chown -R 1001:0 /usr/app/. && chmod -R g+rwX /usr/app/.
USER 1001

ENTRYPOINT ["uv", "run","-m", "alquimia_tech_content_generator_mcp.main"]
