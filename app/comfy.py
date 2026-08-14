import asyncio, json, time, uuid
from pathlib import Path
import httpx

class ComfyError(RuntimeError):
    pass

class ComfyClient:
    def __init__(self, base_url, timeout=1800):
        self.base_url=base_url.rstrip("/")
        self.timeout=timeout

    async def health(self):
        async with httpx.AsyncClient(timeout=10) as c:
            r=await c.get(f"{self.base_url}/system_stats")
            r.raise_for_status()
            return r.json()

    def load_workflow(self, path: Path):
        if not path.exists():
            raise ComfyError(f"Workflow not found: {path}")
        data=json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ComfyError("Workflow must be ComfyUI API-format JSON.")
        return data

    def inject_prompt(self, workflow, prompt, node_id="", field="text"):
        wf=json.loads(json.dumps(workflow))
        if node_id:
            node=wf.get(str(node_id))
            if not node:
                raise ComfyError(f"Prompt node {node_id} not found.")
            node.setdefault("inputs", {})[field]=prompt
            return wf
        candidates=[]
        for nid,node in wf.items():
            if not isinstance(node,dict): continue
            cls=str(node.get("class_type","")).lower()
            for key,val in node.get("inputs",{}).items():
                if isinstance(val,str) and key.lower() in {"text","prompt","positive","positive_prompt"}:
                    score=0 if ("positive" in key.lower() or "cliptext" in cls) else 1
                    candidates.append((score,nid,key))
        if not candidates:
            raise ComfyError("Could not auto-detect a positive prompt field. Set COMFYUI_PROMPT_NODE_ID and COMFYUI_PROMPT_FIELD.")
        candidates.sort()
        _,nid,key=candidates[0]
        wf[nid]["inputs"][key]=prompt
        return wf

    async def queue(self, workflow):
        client_id=str(uuid.uuid4())
        async with httpx.AsyncClient(timeout=30) as c:
            r=await c.post(f"{self.base_url}/prompt", json={"prompt":workflow,"client_id":client_id})
            r.raise_for_status()
            data=r.json()
        if "prompt_id" not in data:
            raise ComfyError(f"ComfyUI rejected workflow: {data}")
        return data["prompt_id"]

    async def wait(self, prompt_id):
        deadline=time.monotonic()+self.timeout
        async with httpx.AsyncClient(timeout=30) as c:
            while time.monotonic()<deadline:
                r=await c.get(f"{self.base_url}/history/{prompt_id}")
                r.raise_for_status()
                item=r.json().get(prompt_id)
                if item:
                    status=item.get("status",{})
                    if status.get("status_str")=="error":
                        raise ComfyError(str(status))
                    outputs=item.get("outputs",{})
                    if outputs:
                        return outputs
                await asyncio.sleep(2)
        raise ComfyError(f"Timed out waiting for ComfyUI job {prompt_id}")

    async def download_first_media(self, outputs, destination: Path):
        candidates=[]
        for out in outputs.values():
            if not isinstance(out,dict): continue
            for key in ("gifs","videos","images"):
                for item in out.get(key,[]) or []:
                    if isinstance(item,dict) and item.get("filename"):
                        candidates.append(item)
        if not candidates:
            raise ComfyError("ComfyUI finished but returned no media output.")
        item=candidates[0]
        params={"filename":item["filename"],"subfolder":item.get("subfolder",""),"type":item.get("type","output")}
        async with httpx.AsyncClient(timeout=120) as c:
            r=await c.get(f"{self.base_url}/view", params=params)
            r.raise_for_status()
            destination.write_bytes(r.content)
        return destination
