#pragma once

namespace Renderer
{
	class Status
	{
	public:
		Status() : m_status(false) {}
		bool HasError() { return m_status; }
	protected:
		bool m_status;
	};
}