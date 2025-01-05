import streamlit as st
import cv2
from ultralytics import YOLO
from helpers import *
import chess
import chess.svg
from io import BytesIO
import base64
from reportlab.pdfgen import canvas
import pandas as pd

# Load your YOLOv8 model
model = YOLO('weights/bestV7.pt')

# Initialize board and move history
if 'board' not in st.session_state:
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.chessboard = [
        ['br', 'bn', 'bb', 'bq', 'bk', 'bb', 'bn', 'br'],
        ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['.', '.', '.', '.', '.', '.', '.', '.'],
        ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp'],
        ['wr', 'wn', 'wb', 'wq', 'wk', 'wb', 'wn', 'wr']
    ]
    st.session_state.previous_board_status = [
        ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black'],
        ['black', 'black', 'black', 'black', 'black', 'black', 'black', 'black'],
        ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white'],
        ['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white']
    ]
    st.session_state.white_moves = pd.DataFrame(columns=["Piece", "From", "To", "Eliminated"])
    st.session_state.black_moves = pd.DataFrame(columns=["Piece", "From", "To", "Eliminated"])

piece_names = {
    'wr': 'Rook', 'wn': 'Knight', 'wb': 'Bishop', 'wq': 'Queen', 'wk': 'King', 'wp': 'Pawn',
    'br': 'Rook', 'bn': 'Knight', 'bb': 'Bishop', 'bq': 'Queen', 'bk': 'King', 'bp': 'Pawn'
}

board = st.session_state.board
move_history = st.session_state.move_history
white_moves = st.session_state.white_moves
black_moves = st.session_state.black_moves

st.title("Chessgame history detection")

st.session_state.conf_threshold = 0.7

warning = st.empty() 
winner_announcement = st.empty()
col1, col2 = st.columns(2)
with col1:
    stframe = st.empty() 
with col2:
    det_out = st.empty()
det_boxes = st.empty()

# Function to render chessboard as SVG
def render_board(board):
    return chess.svg.board(board=board)

# Display board
# Display move tables
white_sec, black_sec, board_sec = st.columns(3)
with white_sec:
    st.write("### White Player Moves")
    white_moves_placeholder = st.dataframe(white_moves)
with black_sec:
    st.write("### Black Player Moves")
    black_moves_placeholder = st.dataframe(black_moves)
with board_sec:
    board_svg_placeholder = st.empty()
def update_board_display():
    board_svg = render_board(board)
    encoded_svg = base64.b64encode(board_svg.encode('utf-8')).decode('utf-8')
    board_svg_placeholder.markdown(f'<img src="data:image/svg+xml;base64,{encoded_svg}" width="400"/>', unsafe_allow_html=True)

update_board_display()

# Detect move from board status
def detect_move(previous_board_status, new_board_status, chessboard):
    move = {}
    for row in range(len(previous_board_status)):
        for col in range(len(previous_board_status[row])):
            if previous_board_status[row][col] != new_board_status[row][col]:
                if new_board_status[row][col] == 'empty' and previous_board_status[row][col] != 'empty':
                    move['start'] = (row, col)
                    move['piece'] = chessboard[row][col]
                elif previous_board_status[row][col] == 'empty' and new_board_status[row][col] != 'empty':
                    move['end'] = (row, col)
                elif previous_board_status[row][col] != new_board_status[row][col]:
                    move['end'] = (row, col)
                    move['eliminated'] = chessboard[row][col]
    return move

# Update chessboard after detecting move
def update_chessboard(move, chessboard):
    start = move['start']
    end = move['end']
    piece = move['piece']
    chessboard[start[0]][start[1]] = '.'
    chessboard[end[0]][end[1]] = piece
    return chessboard

def check_win_condition():
    if board.is_checkmate():
        winner = "Black" if board.turn else "White"  # If it's White's turn and checkmate, Black wins and vice versa
        winner_announcement.success(f"Checkmate! {winner} wins the game!")
        return True
    elif board.is_stalemate():
        winner_announcement.warning("The game is a draw due to stalemate!")
        return True
    elif board.is_insufficient_material():
        winner_announcement.warning("The game is a draw due to insufficient material!")
        return True
    elif board.is_seventyfive_moves():
        winner_announcement.warning("The game is a draw due to the 75-move rule!")
        return True
    elif board.is_fivefold_repetition():
        winner_announcement.warning("The game is a draw due to fivefold repetition!")
        return True
    return False

# Process the frame
def process_frame(frame):
    
    results = model.predict(source=frame, conf=st.session_state.conf_threshold)

    xyxy = results[0].boxes.xyxy
    if not results or len(xyxy) == 0:
        warning.warning(f'No results!')
        return None

    # Perform YOLO prediction
    boxes = results[0].boxes.xyxy.cpu().numpy()
    predicted_classes = results[0].boxes.cls
    class_names = model.names
    predicted_class_names = [class_names[int(cls_idx)] for cls_idx in predicted_classes]    

    if len(xyxy) > 64:
        if st.session_state.conf_threshold < 0.9:
            st.session_state.conf_threshold += 0.05
        det_boxes.write(f'number of detected boxes {len(xyxy)}, while expected is 64. new confidence = {st.session_state.conf_threshold}')
        return
    elif len(xyxy) < 64:
        if st.session_state.conf_threshold > 0.5:
            st.session_state.conf_threshold -= 0.05
        det_boxes.write(f'number of detected boxes {len(xyxy)}, while expected is 64. new confidence = {st.session_state.conf_threshold}')
        return
    det_boxes.write(f"Detected boxes: {len(xyxy)} | Missing cells: {64 - len(xyxy)}")

    detection_vis = results[0].plot() if results else frame
    det_out.image(detection_vis, channels="BGR", use_container_width=True)

    new_board_status = order_detections(boxes, predicted_class_names)

    move = detect_move(st.session_state.previous_board_status, new_board_status, st.session_state.chessboard)

    if 'start' in move and 'end' in move:
        start_square = f"{chr(97 + move['start'][1])}{8 - move['start'][0]}"
        end_square = f"{chr(97 + move['end'][1])}{8 - move['end'][0]}"
        piece_name = piece_names.get(move['piece'], 'Unknown')
        eliminated_piece = piece_names.get(move.get('eliminated', ''), '')

        chess_move = chess.Move.from_uci(f"{start_square}{end_square}")
        move_data = [piece_name, start_square, end_square, eliminated_piece]

        if chess_move in board.legal_moves:
            warning.empty()

            if move['piece'].startswith('w'):
                white_moves.loc[len(white_moves)] = move_data
            else:
                black_moves.loc[len(black_moves)] = move_data

            st.session_state.chessboard = update_chessboard(move, st.session_state.chessboard)
            st.session_state.previous_board_status = new_board_status

            white_moves_placeholder.dataframe(white_moves)
            black_moves_placeholder.dataframe(black_moves)
        
            board.push(chess_move)
            update_board_display()
            check_win_condition()
        else:
            warning.warning(f"move {chess_move} is Illegal move") # here what cause the error

# Export move tables to PDF
def export_to_pdf():
    pdf = BytesIO()
    c = canvas.Canvas(pdf)
    c.setFont("Helvetica", 16)
    c.drawString(200, 800, "Chess Game Move History")
    c.setFont("Helvetica", 14)

    y = 760
    c.drawString(100, y, "White Player Moves:")
    y -= 20
    for index, row in white_moves.iterrows():
        c.drawString(100, y, f"{row['Piece']} from {row['From']} to {row['To']} (Eliminated: {row['Eliminated']})")
        y -= 20

    y -= 40
    c.drawString(100, y, "Black Player Moves:")
    y -= 20
    for index, row in black_moves.iterrows():
        c.drawString(100, y, f"{row['Piece']} from {row['From']} to {row['To']} (Eliminated: {row['Eliminated']})")
        y -= 20

    c.save()
    pdf.seek(0)
    st.download_button(
        label="Download Move History as PDF",
        data=pdf,
        file_name="chess_move_history.pdf",
        mime="application/pdf"
    )

# Live webcam feed
def live_camera_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return

    skip_frame = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                warning.warning("Failed to capture frame. Retrying...")
                continue

            # Display the live video frame
            stframe.image(frame, channels="BGR", use_container_width=True)

            # Process every 10th frame
            if skip_frame % 10 == 0:
                process_frame(frame)

            skip_frame += 1
    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        cap.release()  # Ensure the camera is released when done



# Button to export to PDF
if st.button("Export Move Tables to PDF"):
    export_to_pdf()

live_camera_feed()