import numpy as np

from tests.helpers import make_track
from visionpipe.reid import GlobalIdRegistry, HSVEmbedder, bhattacharyya
from visionpipe.types import Frame


def _person_img(shirt_bgr, pants_bgr=(60, 60, 60), size=(200, 300)):
    img = np.full((size[1], size[0], 3), 128, np.uint8)
    img[50:150, 80:120] = shirt_bgr  # torso region (upper half of the box below)
    img[150:250, 80:120] = pants_bgr
    return img


def _frame(cam, i, img):
    return Frame(cam, i, i / 25, img)


def test_embedding_distance_separates_colours():
    emb = HSVEmbedder()
    box = np.array([80, 50, 120, 250])
    red1, red2 = emb(_person_img((0, 0, 220)), box), emb(_person_img((10, 10, 200)), box)
    blue = emb(_person_img((220, 40, 0)), box)
    assert bhattacharyya(red1, red2) < 0.2 < bhattacharyya(red1, blue)


def test_global_ids_link_same_person_across_cameras():
    reg = GlobalIdRegistry(min_samples=3, sample_every=1, min_age=1)
    red_img, blue_img = _person_img((0, 0, 220)), _person_img((220, 40, 0))

    def track(tid):  # box covers the synthetic person
        t = make_track(tid, 100, 150, w=40, h=200)
        t.age = 10
        return t

    for i in range(6):  # camera A sees red (track 1) and blue (track 2) -- separate crops, one frame each
        reg.update(_frame("camA", i, red_img), [track(1)])
        reg.update(_frame("camA", i, blue_img), [track(2)])
    for i in range(6):  # camera B later sees red as its track 7
        reg.update(_frame("camB", i, red_img), [track(7)])

    a_red, a_blue, b_red = reg.global_id("camA", 1), reg.global_id("camA", 2), reg.global_id("camB", 7)
    assert None not in (a_red, a_blue, b_red)
    assert a_red != a_blue
    assert b_red == a_red
