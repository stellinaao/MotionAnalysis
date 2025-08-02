import cv2
from lib import data

fourcc = cv2.VideoWriter_fourcc(*"XVID")

def save_vidclip(subj_id, sess_id, s_frame, e_frame):
    v_fname = data.get_fname_vid(subj_id, sess_id)
    input_fname = f"data/video/{v_fname}"
    output_fname=f"data/video_clips/{v_fname[:-4]}_f{s_frame}-{e_frame}{v_fname[-4:]}"

    cap = cv2.VideoCapture(input_fname)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)

    if not cap.isOpened():
        raise Exception("ERROR: Could not open video file!")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_fname, fourcc, data.fs, (frame_width, frame_height))
    
    pointer=s_frame
    while pointer < e_frame:
        ret, frame = cap.read()
        if not ret:
            raise Exception("ERROR: End of video or error occurred.")
        out.write(frame)
        pointer+=1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def save_vidclip_frames(subj_id, sess_id, frames, tag=""):
    v_fname = data.get_fname_vid(subj_id, sess_id)
    input_fname = f"data/video/{v_fname}"
    output_fname=f"data/video_clips/{v_fname[:-4]}_{tag}{v_fname[-4:]}"

    cap = cv2.VideoCapture(input_fname)

    if not cap.isOpened():
        raise Exception("ERROR: Could not open video file!")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_fname, fourcc, data.fs, (frame_width, frame_height))

    for frame_id in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            raise Exception("ERROR: End of video or error occurred.")
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()