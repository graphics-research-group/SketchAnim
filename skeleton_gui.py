import cv2
from tkinter import Tk, Canvas, Button, filedialog
from PIL import Image, ImageTk
import os

class SkeletonDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Skeleton Drawer")

        self.canvas = Canvas(root, width=800, height=600)
        self.canvas.pack()

        self.upload_button = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.save_button = Button(root, text="Save Skeleton", command=self.save_skeleton)
        self.save_button.pack()

        self.clear_button = Button(root, text="Clear Skeleton", command=self.clear_skeleton)
        self.clear_button.pack()

        self.image_path = ""
        self.image = None
        self.points = []
        self.jointIndex = []
        self.previousClickonExisting = False
        self.previousClickedIndex = None

        self.canvas.bind("<Button-1>", self.draw_skeleton_at_joint)
        self.canvas.bind("<Button-3>", self.remove_lines)

        self.joint_coordinates = []  
        self.skeleton_edges = []     

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.load_image()

    def load_image(self):
        image = cv2.imread(self.image_path)
        height, width, channels = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(image)
        self.display_image()

    def display_image(self):
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def draw_skeleton_at_joint(self, event):
        x, y = event.x, event.y

        if not self.points:
            # If there are no points, add the first joint
            self.points.append((x, y))
            self.joint_coordinates.append((x, y))
            self.skeleton_edges = self.generate_skeleton_edges()
            self.draw_skeleton()
            return

        # Check if clicking on an existing joint
        for i, (px, py) in enumerate(self.points):
            distance = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if distance <= 5:
                self.previousClickonExisting = True
                self.previousClickedIndex = i
                # self.points = self.points[: i + 1]
                # self.joint_coordinates = self.joint_coordinates[: i + 1]
                # self.jointIndex.append([i , len(self.points)])

                # Draw intermediate skeleton lines between existing joints
                for j in range(i, len(self.points) - 1):
                    self.skeleton_edges.append((j, j + 1))

                self.skeleton_edges = self.generate_skeleton_edges()
                # self.draw_skeleton()
                return

        self.points.append((x, y))
        if self.previousClickonExisting:
            self.jointIndex.append([self.previousClickedIndex, len(self.points) - 1])
            self.previousClickonExisting = False
            self.previousClickedIndex = None
        else:
            self.jointIndex.append([len(self.points) - 2, len(self.points) - 1])
        self.joint_coordinates.append((x, y))

        self.skeleton_edges = self.generate_skeleton_edges()
        self.draw_skeleton()



    # def remove_lines(self, event):
    #     if self.points:
    #         self.points.pop()
    #         self.skeleton_edges = self.generate_skeleton_edges()
    #         self.draw_skeleton()

    def remove_lines(self, event):
        if self.jointIndex:
            removed_joint_index = self.jointIndex.pop()
            self.points.pop()
            self.skeleton_edges = self.generate_skeleton_edges()
            self.draw_skeleton()


    def generate_skeleton_edges(self):
        edges = []
        for i in range(len(self.points) - 1):
            edges.append((i, i + 1))
        return edges

    def draw_skeleton(self):
        if self.image_path:
            image = cv2.imread(self.image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for i in range(len(self.jointIndex)):
                start = self.points[self.jointIndex[i][0]]
                end = self.points[self.jointIndex[i][1]]
                cv2.line(image_rgb, start, end, (255, 0, 0), 2)
                # cv2.line(image_rgb, self.points[i], self.points[i + 1], (255, 0, 0), 2)

            for point in self.points:
                cv2.circle(image_rgb, point, 5, (0, 255, 0), -1)

            self.image = Image.fromarray(image_rgb)
            self.display_image()

    # def clear_skeleton(self):
    #     self.points = []
    #     self.joint_coordinates = []
    #     self.skeleton_edges = []
    #     self.draw_skeleton()

    def clear_skeleton(self):
        self.points = []
        self.joint_coordinates = []
        self.jointIndex = []  # Clear the joint indices as well
        self.skeleton_edges = []
        self.draw_skeleton()


    # def save_skeleton(self):
    #     if self.points:
    #         self.joint_coordinates = [(int(x), int(y)) for x, y in self.points]

    #         with open("joint_coordinates.txt", "w") as f:
    #             for coord in self.joint_coordinates:
    #                 f.write(f"{coord[0]}, {coord[1]}\n")

    #     if self.skeleton_edges:
    #         with open("skeleton_edges.txt", "w") as f:
    #             for edge in self.skeleton_edges:
    #                 f.write(f"{edge[0]} {edge[1]}\n")

    def save_skeleton(self):
        if self.points:
            image = cv2.imread(self.image_path)
            height, width, _ = image.shape

            relative_coordinates = [(x / width, y / height) for x, y in self.points]

            with open("joint_coordinates.txt", "w") as f:
                for coord in relative_coordinates:
                    f.write(f"{coord[0]}, {coord[1]}\n")

        if self.skeleton_edges:
            with open("skeleton_edges.txt", "w") as f:
                for edge in self.skeleton_edges:
                    f.write(f"{edge[0]} {edge[1]}\n")



    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = Tk()
    app = SkeletonDrawer(root)
    app.run()

    