import asyncio

from regulatory_compliance.models.request import AskRequest
from regulatory_compliance.services.query_service import QueryService


async def main():
    request = AskRequest(question="What is Basel III?")

    response = await QueryService.ask_question(request)

    print(response)


asyncio.run(main())
