#pragma once

namespace Renderer
{
	class Status
	{
	public:
		Status() : m_status(false) {}
		bool HasError() { return m_status; }
	protected:
		bool ErrorCheck(bool error)
		{
			m_status = error;
			return HasError();
		}
		bool ErrorCheck(Status& status)
		{
			m_status = status.HasError();
			return HasError();
		}
		bool ErrorCheck(Status* status)
		{
			m_status = status->HasError();
			return HasError();
		}
		bool m_status;
	};
}