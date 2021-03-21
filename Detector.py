import os
import copy

import numpy as np
import cv2
import imutils
import chess

from stockfish import Stockfish




class Detector:

	IMAGE_SIZE = (800, 800)

	max_contour = None
	max_contour_len = 0
	_iter = 0

	@staticmethod
	def find_board(_image):
		
		if Detector._iter >= 60*1/2:
			Detector.max_contour = None
			Detector.max_contour_len = 0
			Detector._iter = 0

		
		image = _image.copy()
		gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

		ret, th = cv2.threshold(gray_image, 127, 255, cv2.THRESH_TRUNC)


		edged_image = cv2.Canny(th, 100, 200, 3)

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
		closed = cv2.morphologyEx(edged_image, cv2.MORPH_CLOSE, kernel)
		edged_image = closed  

		all_contours = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		all_contours = imutils.grab_contours(all_contours)

		
		all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)[:1]
		
		if not all_contours:
			return image, None, None 

		if cv2.contourArea(all_contours[0]) > Detector.max_contour_len:
			Detector.max_contour_len = cv2.contourArea(all_contours[0])
			Detector.max_contour = all_contours[0] 
		
		if Detector.max_contour is None:
			return image, None, None

		perimeter = cv2.arcLength(Detector.max_contour, True)
		ROIdimensions = cv2.approxPolyDP(Detector.max_contour, 0.02*perimeter, True)

		cv2.drawContours(image, [ROIdimensions], -1, (0,255,0), 2)





		if ROIdimensions.shape[0] == 4:
			ROIdimensions = ROIdimensions.reshape(4,2)
		else:
			return image, None, None

		rect = np.zeros((4,2), dtype=np.float32)

		s = np.sum(ROIdimensions, axis=1)
		rect[0] = ROIdimensions[np.argmin(s)]
		rect[2] = ROIdimensions[np.argmax(s)]

		diff = np.diff(ROIdimensions, axis=1)
		rect[1] = ROIdimensions[np.argmin(diff)]
		rect[3] = ROIdimensions[np.argmax(diff)]

		(tl, tr, br, bl) = rect

		widthA = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
		widthB = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
		maxWidth = max(int(widthA), int(widthB))

		heightA = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
		heightB = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
		maxHeight = max(int(heightA), int(heightB))


		dst = np.array([
		    [0,0],
		    [maxWidth-1, 0],
		    [maxWidth-1, maxHeight-1],
		    [0, maxHeight-1]], dtype=np.float32)

		transformMatrix = cv2.getPerspectiveTransform(rect, dst)

		# inversed = np.linalg.inv(transformMatrix)
		inversed = cv2.getPerspectiveTransform(dst, rect)
		# inversed = transformMatrix 
		# print(dst)
		# print()
		# print(rect)
		scan = cv2.warpPerspective(image, transformMatrix, (maxWidth, maxHeight))
		
		return image, scan, inversed


	# @staticmethod
	# def f(_img):
	# 	img = _img.copy()

	# 	for x in range(2, 800-3):
	# 		for y in range(2, 800-3):
	# 			img[x, y] = np.mean(_img[x-1:x+1, y-1:y+1])

	@staticmethod 
	def get_figures(board_img):
		matrix = np.array(["" for i in range(0, 8*8)]).reshape((8,8))

		lines_board = copy.copy(board_img)


		# lines_board = Detector.f(lines_board)

		# for x in range(0, Detector.IMAGE_SIZE[1], 100):
		# 	cv2.line(lines_board, (x, 0), (x, Detector.IMAGE_SIZE[0]), (0, 255, 0), 2)

		# for y in range(0, Detector.IMAGE_SIZE[0], 100):
		# 	cv2.line(lines_board, (0, y), (Detector.IMAGE_SIZE[1], y), (0, 255, 0), 2)

		gray_board = cv2.cvtColor(board_img, cv2.COLOR_RGB2GRAY)
		
		pt1 = cv2.imread("./figures/p1.png")
		pt2 = cv2.imread("./figures/p2.png")
		pt3 = cv2.imread("./figures/p3.png")
		pt4 = cv2.imread("./figures/p4.png")

		rt1 = cv2.imread("./figures/r1.png")
		rt2 = cv2.imread("./figures/r2.png")
		rt3 = cv2.imread("./figures/r3.png")
		rt4 = cv2.imread("./figures/r4.png")

		bt1 = cv2.imread("./figures/b1.png")
		bt2 = cv2.imread("./figures/b2.png")
		bt3 = cv2.imread("./figures/b3.png")
		bt4 = cv2.imread("./figures/b4.png")

		nt1 = cv2.imread("./figures/n1.png")
		nt2 = cv2.imread("./figures/n2.png")
		nt3 = cv2.imread("./figures/n3.png")
		nt4 = cv2.imread("./figures/n4.png")

		qt1 = cv2.imread("./figures/q1.png")
		qt2 = cv2.imread("./figures/q2.png")
		qt3 = cv2.imread("./figures/q3.png")
		qt4 = cv2.imread("./figures/q4.png")

		kt1 = cv2.imread("./figures/k1.png")
		kt2 = cv2.imread("./figures/k2.png")
		kt3 = cv2.imread("./figures/k3.png")
		kt4 = cv2.imread("./figures/k4.png")

		p_templates = [
			("p", pt1, 0.5),
			("p", pt2, 0.7),
			("p", pt3, 0.7),
			("p", pt4, 0.9),

			("r", rt1, 0.5),
			("r", rt2, 0.8),
			("r", rt3, 0.8),
			("r", rt4, 0.9),

			("b", bt1, 0.5),
			("b", bt2, 0.7),
			("b", bt3, 0.75),
			("b", bt4, 0.9),

			("n", nt1, 0.5),
			("n", nt2, 0.7),
			("n", nt3, 0.75),
			("n", nt4, 0.9),

			("q", qt1, 0.5),
			("q", qt2, 0.7),
			("q", qt3, 0.75),
			("q", qt4, 0.9),

			("k", kt1, 0.8),
			("k", kt2, 0.7),
			("k", kt3, 0.9),
			("k", kt4, 0.9),



		]

		ret, wb_board = cv2.threshold(gray_board, 127, 255, cv2.THRESH_BINARY)
		
		

		for name, template, th in p_templates:

			template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
			w, h = template.shape[::-1]
			res = cv2.matchTemplate(gray_board, template, cv2.TM_CCOEFF_NORMED)
			threshold = th


			loc = np.where( res >= threshold )
			for pt in zip(*loc[::-1]):
				cv2.rectangle(lines_board, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
				
				x, y = pt[1] + h//2, pt[0] + w//2
				x += 20

				delta = 5
				# gray_board[x-delta:x+delta+1, y-delta:y+delta+1] = np.zeros((delta*2+1,delta*2+1))
				if not np.mean(gray_board[x-delta:x+delta+1, y-delta:y+delta+1]) < 90:
					name = name.upper()

				matrix[(pt[1] + h // 2) // (Detector.IMAGE_SIZE[1] // 8), (pt[0] + w//2) // (Detector.IMAGE_SIZE[0] // 8)] = name
				

			for x in range(0, 8):
				for y in range(0, 8):
					name = matrix[y, x]
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(lines_board, name, (Detector.IMAGE_SIZE[0]//8*x+10, Detector.IMAGE_SIZE[1]//8*y+30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)


		return matrix, lines_board


	@staticmethod
	def mat_to_fen(arr, _move):
		ans = ""

		for i in range(0, 8):
			for j in range(0, 8):
				if arr[i, j] == "":
					ans += ' '
				else:
					ans += arr[i, j]

			ans += '/'


		ans2 = ''
		c = 0
		if ans[0] == ' ':
			c += 1
		else:
			ans2 += ans[0]

		for i in range(1, len(ans)):
			if ans[i] == ' ' and ans[i-1] == ' ':
				c += 1
			elif ans[i] == ' ' and ans[i-1] != ' ':
				c += 1

			elif ans[i] != ' ' and c == 0:
				ans2 += ans[i]
			elif ans[i] != ' ' and c != 0:
				ans2 += str(c) + ans[i]
				c = 0




		move = _move
		castling = 'KQkq'
		enpassant = '-'
		halfmove  = '0'
		fullmove  = '1'

		return " ".join([ans2[0:len(ans2)-1], move, castling, enpassant, halfmove, fullmove])

	@staticmethod
	def get_from_to_move(maxtrix, image):
		move = input("Чей ход?[w/b]: ").lower()

		board = chess.Board(Detector.mat_to_fen(matrix, move))
		fen = board.fen()

		stockfish = Stockfish("./Engines/stockfish_13_win_x64/stockfish_13_win_x64.exe", 
			parameters={
				"Threads": os.cpu_count(), 
				"Minimum Thinking Time": 30
			}
		)

		stockfish.set_fen_position(fen)
		best_move = stockfish.get_best_move_time(1000)

		print("best move", best_move)
		x1, y1, x2, y2 = Detector.get_from_to(best_move, move)

		# print(x1, y1, x2, y2)
		cv2.rectangle(image, (x1, y1), (x1+100, y1+100), (0,0,255), 2)
		cv2.rectangle(image, (x2, y2), (x2+100, y2+100), (255,0,0), 2)


	@staticmethod
	def get_from_to(move, color):
		x1 = ((ord(move[0]) - ord('a')) ) * 100
		y1 = (8 - (int(move[1]))) * 100
		x2 = ((ord(move[2]) - ord('a')) ) * 100
		y2 = (8 - (int(move[3]))) * 100


		return x1, y1, x2, y2


	@staticmethod
	def back_to_3d(image, w, h, inversed):

		back = cv2.warpPerspective(image, inversed, (h, w))

		return back



class ImgSizeException(Exception):
	pass

def concatenate(img1, img2):
	img = copy.copy(img1)

	if img1.shape[0] != img2.shape[0] or img1.shape[1] != img2.shape[1] or img1.shape[2] != img2.shape[2]:
		raise ImgSizeException

	h, w, d = img1.shape

	# res = np.ones((h, w, d), dtype=np.int32)

	for i in range(0, h):
		for j in range(0, w):
			if np.sum(img2[i, j]) == 0:
				img[i, j] = img1[i, j]
			else:
				for k in range(0, d):
					img[i, j, k] = img2[i, j, k]

	return img

  

if __name__ == "__main__":


	cap = cv2.VideoCapture(0)

	prepared_frame = None

	while(True):
		Detector._iter += 1

		ret, frame = cap.read()


		with_lines, board, inversed = Detector.find_board(frame)
		if board is not None:
			prepared_frame = board
			cv2.imshow("prepared", prepared_frame)

		cv2.imshow("lines", with_lines)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


	cap.release()
	cv2.destroyAllWindows()

	board = prepared_frame

	
	# print("-----------------")
	# frame = cv2.imread("b2.jpg")
	# frame = cv2.resize(frame, (Detector.IMAGE_SIZE[0], Detector.IMAGE_SIZE[1]))
	# with_lines, board, inversed = Detector.find_board(frame)
	# gray_board = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY)
	# ret, gray_board = cv2.threshold(gray_board, 127, 255, cv2.THRESH_BINARY)
	# print(type(frame))

	# print("-------------")


	frame_size = frame.shape

	board_old_w1, board_old_h1, _ = board.shape
	board = cv2.resize(board, (Detector.IMAGE_SIZE[0], Detector.IMAGE_SIZE[1]))

	matrix, debug_board = Detector.get_figures(board)
	# scan = cv2.warpPerspective(board, inversed, (Detector.IMAGE_SIZE[0], Detector.IMAGE_SIZE[1]))


	print(matrix)


	Detector.get_from_to_move(matrix, board)

	board = cv2.resize(board, (board_old_h1, board_old_w1))

	back_img = Detector.back_to_3d(board, frame_size[0], frame_size[1], inversed)


	res_img = concatenate(frame, back_img)

	cv2.imshow('1', res_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()