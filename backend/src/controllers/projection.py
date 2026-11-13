from src.utils import repository as repo
from src.services import projection as service
def create(run_id,data):
    run=repo.get("runs",run_id)
    if data.documents is not None and data.csv_text is not None: raise ValueError("请提供documents数组或csv_text，二选一")
    if data.documents is None and data.csv_text is None: raise ValueError("请提供documents数组或csv_text，二选一")
    return service.project(run,documents=data.documents,csv_text=data.csv_text)
