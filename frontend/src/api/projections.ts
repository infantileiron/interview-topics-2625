import {request} from './client';
export const create=(runId:string,data:unknown)=>request<any>('/runs/'+runId+'/projections','POST',data);
