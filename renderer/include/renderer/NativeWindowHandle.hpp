#pragma once

#include <windows.h>

namespace Renderer
{
	typedef union RendererColor
	{
		float float32[4];
	} RendererColor;

	struct NativeWindowHandle
	{
		NativeWindowHandle(HWND _window, int _width, int _height)
		{
			window = _window;
			width = _width;
			height = _height;
			clear_color = { 0.2f,0.2f,0.2f,1.0f };
		}

		NativeWindowHandle(HWND _window, int _width, int _height, RendererColor _clear_color)
		{
			window = _window;
			width = _width;
			height = _height;
			clear_color = _clear_color;
		}

		HWND window;
		int width;
		int height;
		RendererColor clear_color;
	};
}