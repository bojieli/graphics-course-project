// ConsoleApplication1.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <gl\glut.h>

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

int _tmain(int argc, _TCHAR* argv[])
{
	glutInit(&argc, (char **)argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
	glutInitWindowPosition(50, 100);
	glutInitWindowSize(500, 400);
	glutCreateWindow("直线");
	init();
	glutDisplayFunc(lineSegment);
	glutMainLoop();
	return 0;
}

