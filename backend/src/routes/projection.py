from fastapi import APIRouter,Depends
from src.controllers import projection as controller
from src.schemas.projection import ProjectionInput
from src.services.auth import editor
router=APIRouter(prefix="/runs",tags=["projections"])
@router.post("/{id}/projections",status_code=201)
def create(id:str,data:ProjectionInput,user=Depends(editor)): return controller.create(id,data)
