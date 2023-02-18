const date_format = "mm-dd HH:MM:SS:sss"

function timestamp_logger(logger)
  TransformerLogger(logger) do log
    merge(log, (; message = "$(Dates.format(now(), date_format)) $(log.message)"))
  end
end

function info_filter(logger)
  EarlyFilteredLogger(logger) do log
    log.level == Logging.Info ? true : false
  end
end

function higher_filter(logger)
  EarlyFilteredLogger(logger) do log
    log.level != Logging.Info ? true : false
  end
end


function EvoTradeLogger(io)
  info_logger = io == stdout ? ConsoleLogger() : FormatLogger(io) do io, args
             println(io, args.message)
         end
  higher_logger = ConsoleLogger()

  TeeLogger(
    higher_filter(higher_logger),
    info_filter(timestamp_logger(info_logger))
  )
end
