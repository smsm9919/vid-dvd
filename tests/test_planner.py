from app.planner import local_scene_plan

def test_scene_count_and_duration():
    scenes=local_scene_plan("lion vs snake","A lion meets a snake. They fight.",30,5)
    assert len(scenes)==5
    assert abs(sum(s.duration for s in scenes)-30)<0.1

def test_prompt_is_cinematic():
    scenes=local_scene_plan("test","Hello world.",10,2)
    assert "cinematic" in scenes[0].prompt.lower()
