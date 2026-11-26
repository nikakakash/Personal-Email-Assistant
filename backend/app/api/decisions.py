"""`POST /proposals/{id}/decision`: a thin wrapper over `decide_proposal`.

`decide_proposal` commits or rolls back the session it is given (and may roll back more than
once), so it gets a dedicated session rather than the request-scoped one.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends

from app.actions.decisions import DecisionIn, DecisionOutcome, decide_proposal
from app.api.deps import OWNER_ONLY, OwnerId, SessionmakerDep, get_mail
from app.providers.mock_mail import MockMailProvider

router = APIRouter(prefix="/api/v1/proposals", tags=["decisions"], dependencies=OWNER_ONLY)


@router.post("/{proposal_id}/decision", status_code=202)
async def decision(
    proposal_id: uuid.UUID,
    body: DecisionIn,
    owner_id: OwnerId,
    maker: SessionmakerDep,
    mail: Annotated[MockMailProvider, Depends(get_mail)],
) -> DecisionOutcome:
    async with maker() as session:
        return await decide_proposal(session, owner_id, proposal_id, body, mail=mail)
