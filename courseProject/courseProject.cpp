#include <GL/glut.h>

void init(void)
{
	glClearColor(1.0, 1.0, 1.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, 200.0, 0.0, 200.0);
}

void lineSegment(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 0.0, 0.0);
	glBegin(GL_LINES);
		glVertex2i(180,15);
		glVertex2i(10,145);
	glEnd();
	glFlush();
}

int main(int argc, char** argv)
{
	glutInit(&argc, (char **)argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
	glutInitWindowPosition(50, 100);
	glutInitWindowSize(500, 400);
	glutCreateWindow("Ö±Ïß");
	init();
	glutDisplayFunc(lineSegment);
	glutMainLoop();
	return 0;
}

