"""Tool definitions for the plant operations copilot.

Each tool is a Pydantic model. The class docstring becomes the tool description
shown to the model, and each Field description documents one argument. Pydantic
also gives us strict validation, which the evaluation harness reuses to check
whether a predicted tool call is valid.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

MachineId = Annotated[
    str, Field(pattern=r"^M-\d{3}$", description="Machine identifier, e.g. 'M-104'.")
]
LineId = Annotated[
    str, Field(pattern=r"^L[1-6]$", description="Production line identifier, from 'L1' to 'L6'.")
]
PartNumber = Annotated[
    str, Field(pattern=r"^P-\d{4}$", description="Spare part number, e.g. 'P-2231'.")
]

Team = Literal["mechanical", "electrical", "automation", "quality"]
Priority = Literal["low", "medium", "high", "critical"]
Sensor = Literal["temperature", "vibration", "pressure", "current", "speed"]


class Tool(BaseModel):
    """Base class for every tool. Subclasses must set `name`."""

    model_config = ConfigDict(extra="forbid")

    name: ClassVar[str]


class GetMachineStatus(Tool):
    """Get the current state of a machine: running, idle, stopped or in alarm,
    with its active alarms if any."""

    name: ClassVar[str] = "get_machine_status"

    machine_id: MachineId


class GetSensorReadings(Tool):
    """Get recent readings from one sensor of a machine over a time window."""

    name: ClassVar[str] = "get_sensor_readings"

    machine_id: MachineId
    sensor: Sensor = Field(description="Which sensor to read.")
    window_minutes: int = Field(
        default=60, ge=1, le=1440, description="How far back to look, in minutes (1 to 1440)."
    )


class GetFailureRisk(Tool):
    """Get the predicted probability that a machine fails within a time horizon,
    with the main contributing signals."""

    name: ClassVar[str] = "get_failure_risk"

    machine_id: MachineId
    horizon_hours: int = Field(
        default=24, ge=1, le=168, description="Prediction horizon in hours (1 to 168)."
    )


class CreateMaintenanceTicket(Tool):
    """Open a maintenance ticket for a machine and assign it to a team."""

    name: ClassVar[str] = "create_maintenance_ticket"

    machine_id: MachineId
    issue: str = Field(
        min_length=5, max_length=300, description="Short description of the problem."
    )
    priority: Priority = Field(description="Ticket priority.")
    team: Team = Field(description="Team responsible for the fix.")


class UpdateTicketStatus(Tool):
    """Change the status of an existing maintenance ticket, optionally with a note."""

    name: ClassVar[str] = "update_ticket_status"

    ticket_id: str = Field(
        pattern=r"^TCK-\d{5}$", description="Ticket identifier, e.g. 'TCK-00412'."
    )
    status: Literal["open", "in_progress", "resolved", "closed"] = Field(description="New status.")
    note: str | None = Field(default=None, max_length=300, description="Optional comment.")


class ScheduleDowntime(Tool):
    """Plan a downtime window for a machine (maintenance, cleaning, changeover)."""

    name: ClassVar[str] = "schedule_downtime"

    machine_id: MachineId
    start_time: datetime = Field(
        description="Start of the downtime, ISO 8601 (e.g. 2026-10-21T14:00)."
    )
    duration_minutes: int = Field(ge=5, le=2880, description="Length of the downtime in minutes.")
    reason: str = Field(min_length=3, max_length=200, description="Why the machine is stopped.")


class GetProductionKpis(Tool):
    """Get a production KPI for a line over a period."""

    name: ClassVar[str] = "get_production_kpis"

    line_id: LineId
    kpi: Literal["oee", "throughput", "scrap_rate", "availability"] = Field(
        description="Which KPI to compute."
    )
    period: Literal["shift", "day", "week"] = Field(
        default="day", description="Aggregation period."
    )


class NotifyTeam(Tool):
    """Send a message to a team's on-call channel."""

    name: ClassVar[str] = "notify_team"

    team: Team = Field(description="Team to notify.")
    message: str = Field(min_length=5, max_length=500, description="Message to send.")
    urgency: Literal["info", "warning", "critical"] = Field(
        default="info", description="Urgency level."
    )


class GetSparePartStock(Tool):
    """Check how many units of a spare part are in stock, and where."""

    name: ClassVar[str] = "get_spare_part_stock"

    part_number: PartNumber


class OrderSparePart(Tool):
    """Order a spare part from the supplier."""

    name: ClassVar[str] = "order_spare_part"

    part_number: PartNumber
    quantity: int = Field(ge=1, le=500, description="Number of units to order.")
    urgent: bool = Field(default=False, description="True for express delivery.")


class GetShiftReport(Tool):
    """Get the end-of-shift report of a production line: output, stops and incidents."""

    name: ClassVar[str] = "get_shift_report"

    line_id: LineId
    shift_date: date = Field(description="Date of the shift, ISO 8601 (e.g. 2026-10-20).")
    shift: Literal["morning", "afternoon", "night"] = Field(description="Which shift.")


class SearchMaintenanceHistory(Tool):
    """Search past maintenance interventions on a machine."""

    name: ClassVar[str] = "search_maintenance_history"

    machine_id: MachineId
    keyword: str | None = Field(
        default=None, max_length=100, description="Optional keyword, e.g. 'bearing' or 'leak'."
    )
    limit: int = Field(default=5, ge=1, le=50, description="Maximum number of results.")


ALL_TOOLS: tuple[type[Tool], ...] = (
    GetMachineStatus,
    GetSensorReadings,
    GetFailureRisk,
    CreateMaintenanceTicket,
    UpdateTicketStatus,
    ScheduleDowntime,
    GetProductionKpis,
    NotifyTeam,
    GetSparePartStock,
    OrderSparePart,
    GetShiftReport,
    SearchMaintenanceHistory,
)
